from utils import ( 
  imsave,
  prepare_data
)

import time
import os
import matplotlib.pyplot as plt
import re
import numpy as np
import tensorflow as tf
import scipy.io as scio
from ops import *
from ssim import *
from utils import *
# local library
import rgb_lab_formulation as Conv_img
import vgg
class T_CNN(object):

  def __init__(self, 
               sess, 
               image_height=128,
               image_width=128,
               label_height=128, 
               label_width=128,
               batch_size=4,
               c_dim=3, 
               checkpoint_dir=None, 
               sample_dir=None
               ):

    self.sess = sess
    self.is_grayscale = (c_dim == 1)
    self.image_height = image_height
    self.image_width = image_width
    self.label_height = label_height
    self.label_width = label_width
    self.dropout_keep_prob=0.5
    self.batch_size = batch_size
    self.c_dim = c_dim
    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    # 
    self.df_dim = 64
    self.vgg_dir='vgg_pretrained/imagenet-vgg-verydeep-19.mat'
    self.CONTENT_LAYER = 'relu5_4'
    self.build_model()

  def build_model(self):
    self.images       = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images')
    self.labels_image = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='labels_image')
    self._depth = tf.placeholder(tf.float32, [self.batch_size, self.label_height, self.label_width, 1], name='depth_train')

  
    self.images_test       = tf.placeholder(tf.float32, [1, self.image_height, self.image_width, self.c_dim], name='images_test')
    self.labels_test_image = tf.placeholder(tf.float32, [1,self.label_height,self.label_width, self.c_dim], name='labels_test_image')
    self._test_depth = tf.placeholder(tf.float32, [1,self.label_height,self.label_width, 1], name='test_depth')


    self.raw= self.model()
    t_vars = tf.trainable_variables()
    self.fusion_var = [var for var in t_vars if 'fusion' in var.name]
    self.saver = tf.train.Saver()


    
    self.labels_texture_vgg = vgg.net(self.vgg_dir, vgg.preprocess(self.labels_image* 255))
    self.raw_texture_vgg = vgg.net(self.vgg_dir, vgg.preprocess(self.raw * 255))
    self.loss_vgg_raw =0.05*tf.reduce_mean(tf.square(self.raw_texture_vgg[self.CONTENT_LAYER]-self.labels_texture_vgg[self.CONTENT_LAYER]))

    self.MSE=5*tf.reduce_mean(tf.square(self.labels_image-self.raw))



    self.loss = self.MSE+self.loss_vgg_raw 
    #t_vars = tf.trainable_variables()

    self.saver = tf.train.Saver(max_to_keep=0)
    
  def train(self, config):
    if config.is_train:     
      data_train_list   = prepare_data(self.sess, dataset="input_train")
      image_train_list  = prepare_data(self.sess, dataset="gt_train")
      depth_train_list = prepare_data(self.sess, dataset="depth_train")


      data_test_list = prepare_data(self.sess, dataset="input_test")
      image_test_list = prepare_data(self.sess, dataset="gt_test")
      depth_test_list = prepare_data(self.sess, dataset="depth_test")


      seed = 1024
      np.random.seed(seed)
      np.random.shuffle(data_train_list)
      np.random.seed(seed)
      np.random.shuffle(image_train_list)
      np.random.seed(seed)
      np.random.shuffle(depth_train_list)


    else:
      data_test_list = prepare_data(self.sess, dataset="input_test")
      image_test_list = prepare_data(self.sess, dataset="gt_test")
      depth_test_list = prepare_data(self.sess, dataset="depth_test")



    sample_data_files = data_test_list[16:20]
    sample_image_files = image_test_list[16:20]
    sample_depth_files = depth_test_list[16:20]



    sample_data_pre = [
          get_image(sample_data_file,
                    is_grayscale=self.is_grayscale) for sample_data_file in sample_data_files]
    sample_lable_image_pre = [
          get_image(sample_image_file,
                    is_grayscale=self.is_grayscale) for sample_image_file in sample_image_files]
    sample_depth_pre = [
          get_image(sample_depth_file,
                    is_grayscale=self.is_grayscale) for sample_depth_file in sample_depth_files]

    sample_inputs_data_pre = np.array(sample_data_pre).astype(np.float32)
    sample_inputs_lable_image_pre = np.array(sample_lable_image_pre).astype(np.float32)
    sample_inputs_depth_pre = np.array(sample_depth_pre).astype(np.float32)[:, :, :, None]


#################################################################################################
    x_offset = np.random.randint(low=0, high=330, size=1)[0]
    y_offset = np.random.randint(low=0, high=490, size=1)[0]
  
    sample_inputs_data=random_crop_and_flip_3(sample_inputs_data_pre, x_offset,y_offset,128,128)
    sample_inputs_lable_image=random_crop_and_flip_3(sample_inputs_lable_image_pre, x_offset,y_offset,128,128)
    sample_inputs_depth=random_crop_and_flip_1(sample_inputs_depth_pre,  x_offset,y_offset,128,128)

    self.g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
             .minimize(self.loss,var_list=self.fusion_var)
               
    tf.global_variables_initializer().run()
    
    
    counter = 0
    start_time = time.time()

    if self.load(self.checkpoint_dir): 
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    if config.is_train:
      print("Training...")
      loss = np.ones(config.epoch)

      for ep in range(config.epoch):
        batch_idxs = len(data_train_list) // config.batch_size
        for idx in range(0, batch_idxs):

          batch_files       = data_train_list[idx*config.batch_size:(idx+1)*config.batch_size]
          batch_image_files = image_train_list[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_depth_files = depth_train_list[idx*config.batch_size : (idx+1)*config.batch_size]


          batch_ = [
          get_image(batch_file,
                    is_grayscale=self.is_grayscale) for batch_file in batch_files]
          batch_labels_image = [
          get_image(batch_image_file,
                    is_grayscale=self.is_grayscale) for batch_image_file in batch_image_files]
          batch_depth_image = [
          get_image(batch_depth_file,
                    is_grayscale=self.is_grayscale) for batch_depth_file in batch_depth_files]

          batch_input_pre = np.array(batch_).astype(np.float32)
          batch_image_input_pre = np.array(batch_labels_image).astype(np.float32)
          batch_depth_pre = np.array(batch_depth_image).astype(np.float32)[:, :, :, None]


 #################################################################################################
          x_offset = np.random.randint(low=0, high=330, size=1)[0]
          y_offset = np.random.randint(low=0, high=490, size=1)[0]
 
          batch_input=random_crop_and_flip_3(batch_input_pre, x_offset,y_offset,128,128)
          batch_image_input=random_crop_and_flip_3(batch_image_input_pre, x_offset,y_offset,128,128)
          batch_depth_input=random_crop_and_flip_1(batch_depth_pre,  x_offset,y_offset,128,128)

          counter += 1
          _, err1,err2,err3= self.sess.run([self.g_optim, self.MSE, self.loss_vgg_raw, self.loss], feed_dict={self.images: batch_input, self.labels_image:batch_image_input,self._depth:batch_depth_input}) 
          


          if counter % 100 == 0:
            print("Epoch: [%2d], step: [%2d], time: [%4.4f],mse_loss: [%.8f],vgg_loss: [%.8f],final_loss: [%.8f]" \
              % ((ep+1), counter, time.time()-start_time,err1,err2,err3))
          if idx  == batch_idxs-1:
            batch_test_idxs = len(data_test_list) // config.batch_size
            err_test =  np.ones(batch_test_idxs)
            for idx_test in range(0,batch_test_idxs):

              sample_data_files = data_test_list[idx_test*config.batch_size:(idx_test+1)*config.batch_size]
              sample_image_files = image_test_list[idx_test*config.batch_size : (idx_test+1)*config.batch_size]
              sample_image_files1 = depth_test_list[idx_test*config.batch_size : (idx_test+1)*config.batch_size]

              sample_data = [get_image(sample_data_file,
                            is_grayscale=self.is_grayscale) for sample_data_file in sample_data_files]
              sample_lable_image = [get_image(sample_image_file,
                                    is_grayscale=self.is_grayscale) for sample_image_file in sample_image_files]
              sample_lable_image1 = [get_image(sample_image_file1,
                                    is_grayscale=self.is_grayscale) for sample_image_file1 in sample_image_files1]

              sample_inputs_data_pre = np.array(sample_data).astype(np.float32)
              sample_inputs_lable_image_pre = np.array(sample_lable_image).astype(np.float32)
              sample_inputs_lable_image1_pre = np.array(sample_lable_image1).astype(np.float32)[:, :, :, None]

 #################################################################################################
              x_offset = np.random.randint(low=0, high=330, size=1)[0]
              y_offset = np.random.randint(low=0, high=490, size=1)[0]
              sample_inputs_data=random_crop_and_flip_3(sample_inputs_data_pre, x_offset,y_offset,128,128)
              sample_inputs_lable_image=random_crop_and_flip_3(sample_inputs_lable_image_pre, x_offset,y_offset,128,128)
              sample_inputs_lable_image1=random_crop_and_flip_1(sample_inputs_lable_image1_pre,  x_offset,y_offset,128,128)

              err_test[idx_test] = self.sess.run(self.loss, feed_dict={self.images: sample_inputs_data,  self.labels_image:sample_inputs_lable_image,self._depth:sample_inputs_lable_image1})    
            loss[ep]=np.mean(err_test)
            print(loss)
            self.save(config.checkpoint_dir, counter)


  def model(self):

    with tf.variable_scope("fusion_branch") as scope1:

      depth_2down=max_pool_2x2(1-self._depth) 
      depth_4down=max_pool_2x2(depth_2down) 
      depth_8down=max_pool_2x2(depth_4down) 
# first HSV encoder
      HSV=tf.image.rgb_to_hsv(self.images)
      conv2_1_HSV = tf.nn.relu(conv2d(HSV, 3,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_1_HSV"))
      conv2_cb1_1_HSV = tf.nn.relu(conv2d(conv2_1_HSV, 3,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb1_1_HSV"))
      conv2_cb1_2_HSV = tf.nn.relu(conv2d(conv2_cb1_1_HSV, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb1_2_HSV"))
      conv2_cb1_3_HSV = conv2d(conv2_cb1_2_HSV, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb1_3_HSV")
      first_add_HSV=tf.add(conv2_1_HSV,conv2_cb1_3_HSV)
      conv2_2_HSV = tf.nn.relu(conv2d(first_add_HSV, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_2_HSV"))     
      conv2_cb1_4_HSV = tf.nn.relu(conv2d(conv2_2_HSV, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb1_4_HSV"))
      conv2_cb1_5_HSV = tf.nn.relu(conv2d(conv2_cb1_4_HSV, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb1_5_HSV"))
      conv2_cb1_6_HSV =conv2d(conv2_cb1_5_HSV, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb1_6_HSV")
      first_add1_HSV=tf.add(conv2_2_HSV,conv2_cb1_6_HSV)
      encoder1_down2_HSV=max_pool_2x2(first_add1_HSV)      

# first YUV encoder
      YUV=Conv_img.rgb_to_lab(self.images)
      conv2_1_YUV = tf.nn.relu(conv2d(YUV, 3,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_1_YUV"))     
      conv2_cb1_1_YUV = tf.nn.relu(conv2d(conv2_1_YUV, 3,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb1_1_YUV"))
      conv2_cb1_2_YUV = tf.nn.relu(conv2d(conv2_cb1_1_YUV, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb1_2_YUV"))
      conv2_cb1_3_YUV = conv2d(conv2_cb1_2_YUV, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb1_3_YUV")
      first_add_YUV=tf.add(conv2_1_YUV,conv2_cb1_3_YUV)
      conv2_2_YUV = tf.nn.relu(conv2d(first_add_YUV, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_2_YUV"))
      conv2_cb1_4_YUV = tf.nn.relu(conv2d(conv2_2_YUV, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb1_4_YUV"))
      conv2_cb1_5_YUV = tf.nn.relu(conv2d(conv2_cb1_4_YUV, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb1_5_YUV"))
      conv2_cb1_6_YUV =conv2d(conv2_cb1_5_YUV, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb1_6_YUV")
      first_add1_YUV=tf.add(conv2_2_YUV,conv2_cb1_6_YUV)
      encoder1_down2_YUV=max_pool_2x2(first_add1_YUV)   

# first RGB encoder
      conv2_1 = tf.nn.relu(conv2d(self.images, 3,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_1"))
      conv2_cb1_1 = tf.nn.relu(conv2d(tf.concat(axis = 3, values = [conv2_1,conv2_1_HSV,conv2_1_YUV]), 128, 128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb1_1"))
      conv2_cb1_2 = tf.nn.relu(conv2d(tf.concat(axis = 3, values = [conv2_cb1_1,conv2_cb1_1_HSV,conv2_cb1_1_YUV]), 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb1_2"))
      conv2_cb1_3 = conv2d(tf.concat(axis = 3, values = [conv2_cb1_2,conv2_cb1_3_HSV,conv2_cb1_3_YUV]), 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb1_3")
      first_add=tf.add(conv2_1,conv2_cb1_3)
      conv2_2 = tf.nn.relu(conv2d(tf.concat(axis = 3, values = [first_add,first_add_HSV,first_add_YUV]), 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_2"))
      conv2_cb1_4 = tf.nn.relu(conv2d(tf.concat(axis = 3, values = [conv2_2,conv2_2_HSV,conv2_2_YUV]), 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb1_4"))
      conv2_cb1_5 = tf.nn.relu(conv2d(tf.concat(axis = 3, values = [conv2_cb1_4,conv2_cb1_4_HSV,conv2_cb1_4_YUV]), 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb1_5"))
      conv2_cb1_6 = conv2d(tf.concat(axis = 3, values = [conv2_cb1_5,conv2_cb1_5_HSV,conv2_cb1_5_YUV]), 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb1_6")
      first_add1=tf.add(conv2_2,conv2_cb1_6)
      encoder1_down2=max_pool_2x2(first_add1)     


# second hsv encoder
      conv2_2_1_HSV = tf.nn.relu(conv2d(encoder1_down2_HSV, 3,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_2_1_HSV"))
      conv2_cb2_1_HSV = tf.nn.relu(conv2d(conv2_2_1_HSV, 3,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb2_1_HSV"))
      conv2_cb2_2_HSV = tf.nn.relu(conv2d(conv2_cb2_1_HSV, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb2_2_HSV"))
      conv2_cb2_3_HSV = conv2d(conv2_cb2_2_HSV, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb2_3_HSV")
      second_add_HSV=tf.add(conv2_2_1_HSV,conv2_cb2_3_HSV)
      conv2_2_2_HSV = tf.nn.relu(conv2d(second_add_HSV, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_2_2_HSV"))
      conv2_cb2_4_HSV = tf.nn.relu(conv2d(conv2_2_2_HSV, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb2_4_HSV"))
      conv2_cb2_5_HSV = tf.nn.relu(conv2d(conv2_cb2_4_HSV, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb2_5_HSV"))
      conv2_cb2_6_HSV = conv2d(conv2_cb2_5_HSV, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb2_6_HSV")
      second_add1_HSV=tf.add(conv2_2_2_HSV,conv2_cb2_6_HSV)
      encoder2_down2_HSV=max_pool_2x2(second_add1_HSV)    

# second YUV encoder
      conv2_2_1_YUV = tf.nn.relu(conv2d(encoder1_down2_YUV, 3,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_2_1_YUV"))
      conv2_cb2_1_YUV = tf.nn.relu(conv2d(conv2_2_1_YUV, 3,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb2_1_YUV"))
      conv2_cb2_2_YUV = tf.nn.relu(conv2d(conv2_cb2_1_YUV, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb2_2_YUV"))
      conv2_cb2_3_YUV = conv2d(conv2_cb2_2_YUV, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb2_3_YUV")
      second_add_YUV=tf.add(conv2_2_1_YUV,conv2_cb2_3_YUV)
      conv2_2_2_YUV = tf.nn.relu(conv2d(second_add_YUV, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_2_2_YUV"))
      conv2_cb2_4_YUV = tf.nn.relu(conv2d(conv2_2_2_YUV, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb2_4_YUV"))
      conv2_cb2_5_YUV = tf.nn.relu(conv2d(conv2_cb2_4_YUV, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb2_5_YUV"))
      conv2_cb2_6_YUV = conv2d(conv2_cb2_5_YUV, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb2_6_YUV")
      second_add1_YUV=tf.add(conv2_2_2_YUV,conv2_cb2_6_YUV)
      encoder2_down2_YUV=max_pool_2x2(second_add1_YUV)    



# second RGB encoder
      conv2_2_1 = tf.nn.relu(conv2d(encoder1_down2, 3,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_2_1"))
      conv2_cb2_1 = tf.nn.relu(conv2d(tf.concat(axis = 3, values = [conv2_2_1,conv2_2_1_HSV,conv2_2_1_YUV]), 3,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb2_1"))
      conv2_cb2_2 = tf.nn.relu(conv2d(tf.concat(axis = 3, values = [conv2_cb2_1,conv2_cb2_1_HSV,conv2_cb2_1_YUV]), 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb2_2"))
      conv2_cb2_3 = conv2d(tf.concat(axis = 3, values = [conv2_cb2_2,conv2_cb2_3_HSV,conv2_cb2_3_YUV]), 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb2_3")
      second_add=tf.add(conv2_2_1,conv2_cb2_3)
      conv2_2_2 = tf.nn.relu(conv2d(tf.concat(axis = 3, values = [second_add,second_add_HSV,second_add_YUV]), 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_2_2"))
      conv2_cb2_4 = tf.nn.relu(conv2d(tf.concat(axis = 3, values = [conv2_2_2,conv2_2_2_HSV,conv2_2_2_YUV]), 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb2_4"))
      conv2_cb2_5 = tf.nn.relu(conv2d(tf.concat(axis = 3, values = [conv2_cb2_4,conv2_cb2_4_HSV,conv2_cb2_4_YUV]), 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb2_5"))
      conv2_cb2_6 = conv2d(tf.concat(axis = 3, values = [conv2_cb2_5,conv2_cb2_5_HSV,conv2_cb2_5_YUV]), 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb2_6")
      second_add1=tf.add(conv2_2_2,conv2_cb2_6)
      encoder2_down2=max_pool_2x2(second_add1)    



# third hsv encoder
      conv2_3_1_HSV = tf.nn.relu(conv2d(encoder2_down2_HSV, 3,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_3_1_HSV"))
      conv2_cb3_1_HSV = tf.nn.relu(conv2d(conv2_3_1_HSV, 3,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb3_1_HSV"))
      conv2_cb3_2_HSV = tf.nn.relu(conv2d(conv2_cb3_1_HSV, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb3_2_HSV"))
      conv2_cb3_3_HSV = conv2d(conv2_cb3_2_HSV, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb3_3_HSV")
      third_add_HSV=tf.add(conv2_3_1_HSV,conv2_cb3_3_HSV)
      conv2_3_2_HSV = tf.nn.relu(conv2d(third_add_HSV, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_3_2_HSV"))
      conv2_cb3_4_HSV = tf.nn.relu(conv2d(conv2_3_2_HSV, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb3_4_HSV"))
      conv2_cb3_5_HSV = tf.nn.relu(conv2d(conv2_cb3_4_HSV, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb3_5_HSV"))
      conv2_cb3_6_HSV = conv2d(conv2_cb3_5_HSV, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb3_6_HSV")
      third_add1_HSV=tf.add(conv2_3_2_HSV,conv2_cb3_6_HSV)

# third YUV encoder
      conv2_3_1_YUV = tf.nn.relu(conv2d(encoder2_down2_YUV, 3,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_3_1_YUV"))
      conv2_cb3_1_YUV = tf.nn.relu(conv2d(conv2_3_1_YUV, 3,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb3_1_YUV"))
      conv2_cb3_2_YUV = tf.nn.relu(conv2d(conv2_cb3_1_YUV, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb3_2_YUV"))
      conv2_cb3_3_YUV = conv2d(conv2_cb3_2_YUV, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb3_3_YUV")
      third_add_YUV=tf.add(conv2_3_1_YUV,conv2_cb3_3_YUV)
      conv2_3_2_YUV = tf.nn.relu(conv2d(third_add_YUV, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_3_2_YUV"))
      conv2_cb3_4_YUV = tf.nn.relu(conv2d(conv2_3_2_YUV, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb3_4_YUV"))
      conv2_cb3_5_YUV = tf.nn.relu(conv2d(conv2_cb3_4_YUV, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb3_5_YUV"))
      conv2_cb3_6_YUV =conv2d(conv2_cb3_5_YUV, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb3_6_YUV")
      third_add1_YUV=tf.add(conv2_3_2_YUV,conv2_cb3_6_YUV)


# third RGB encoder
      conv2_3_1 = tf.nn.relu(conv2d(encoder2_down2, 3,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_3_1"))
      conv2_cb3_1 = tf.nn.relu(conv2d(tf.concat(axis = 3, values = [conv2_3_1,conv2_3_1_HSV,conv2_3_1_YUV]), 3,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb3_1"))
      conv2_cb3_2 = tf.nn.relu(conv2d(tf.concat(axis = 3, values = [conv2_cb3_1,conv2_cb3_1_HSV,conv2_cb3_1_YUV]), 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb3_2"))
      conv2_cb3_3 = conv2d(tf.concat(axis = 3, values = [conv2_cb3_2,conv2_cb3_2_HSV,conv2_cb3_2_YUV]), 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb3_3")
      third_add=tf.add(conv2_3_1,conv2_cb3_3)
      conv2_3_2 = tf.nn.relu(conv2d(tf.concat(axis = 3, values = [third_add,third_add_HSV,third_add_YUV]), 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_3_2"))
      conv2_cb3_4 = tf.nn.relu(conv2d(tf.concat(axis = 3, values = [conv2_3_2,conv2_3_2_HSV,conv2_3_2_YUV]), 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb3_4"))
      conv2_cb3_5 = tf.nn.relu(conv2d(tf.concat(axis = 3, values = [conv2_cb3_4,conv2_cb3_4_HSV,conv2_cb3_4_YUV]), 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb3_5"))
      conv2_cb3_6 =conv2d(tf.concat(axis = 3, values = [conv2_cb3_5,conv2_cb3_5_HSV,conv2_cb3_5_YUV]), 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb3_6")
      third_add1=tf.add(conv2_3_2,conv2_cb3_6)

      
########### concate
      third_con= tf.concat(axis = 3, values = [third_add1,third_add1_HSV,third_add1_YUV])
      channle_weight_third_con_temp=self.Squeeze_excitation_layer(third_con, out_dim=1536, ratio=16, layer_name='channle_weight_third_con_temp')
      third_con_ff = tf.nn.relu(conv2d(channle_weight_third_con_temp, 1536,512,k_h=3, k_w=3, d_h=1, d_w=1,name="third_con_ff"))

      second_con= tf.concat(axis = 3, values = [second_add1,second_add1_HSV,second_add1_YUV]) 
      channle_weight_second_con_temp=self.Squeeze_excitation_layer(second_con, out_dim=768, ratio=16, layer_name='channle_weight_second_con_temp')
      second_con_ff = tf.nn.relu(conv2d(channle_weight_second_con_temp, 768,256,k_h=3, k_w=3, d_h=1, d_w=1,name="second_con_ff"))

      first_con= tf.concat(axis = 3, values = [first_add1,first_add1_HSV,first_add1_YUV]) 
      channle_weight_first_con_temp=self.Squeeze_excitation_layer(first_con, out_dim=384, ratio=16, layer_name='channle_weight_first_con_temp')
      first_con_ff = tf.nn.relu(conv2d(channle_weight_first_con_temp, 384,128,k_h=3, k_w=3, d_h=1, d_w=1,name="first_con_ff"))
#############################################################################################################
#first decoder
      decoder_input=tf.add(third_con_ff,tf.multiply(third_con_ff,depth_4down))
      conv2_1_1_dc = tf.nn.relu(conv2d(decoder_input, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_1_1_dc"))
      conv2_dc1_1 = tf.nn.relu(conv2d(conv2_1_1_dc, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_dc1_1"))
      conv2_dc1_2 = tf.nn.relu(conv2d(conv2_dc1_1, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_dc1_2"))
      conv2_dc1_3 = conv2d(conv2_dc1_2, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_dc1_3")
      first_dadd=tf.add(conv2_1_1_dc,conv2_dc1_3)
      conv2_1_2_dc = tf.nn.relu(conv2d(first_dadd, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_1_2_dc"))
      conv2_dc1_4 = tf.nn.relu(conv2d(conv2_1_2_dc, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_dc1_4"))
      conv2_dc1_5 = tf.nn.relu(conv2d(conv2_dc1_4, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_dc1_5"))
      conv2_dc1_6 = conv2d(conv2_dc1_5, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_dc1_6")
      first_dadd1=tf.add(conv2_1_2_dc,conv2_dc1_6)
      decoder1_down2=tf.image.resize_bilinear(first_dadd1,[second_con_ff.get_shape().as_list()[1], second_con_ff.get_shape().as_list()[2]])
     
# second decoder
      decoder_input1=tf.add(second_con_ff,tf.multiply(second_con_ff,depth_2down))
      concate_1 = tf.concat(axis = 3, values = [decoder1_down2,decoder_input1])  
      conv2_1_3_dc = tf.nn.relu(conv2d(concate_1, 512,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_1_3_dc"))
      conv2_dc2_1 = tf.nn.relu(conv2d(conv2_1_3_dc, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_dc2_1"))      
      conv2_dc2_2 = tf.nn.relu(conv2d(conv2_dc2_1, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_dc2_2"))
      conv2_dc2_3 = conv2d(conv2_dc2_2, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_dc2_3")
      second_dadd=tf.add(conv2_1_3_dc,conv2_dc2_3)
      conv2_1_4_dc = tf.nn.relu(conv2d(second_dadd, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_1_4_dc"))
      conv2_dc2_4 = tf.nn.relu(conv2d(conv2_1_4_dc, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_dc2_4"))
      conv2_dc2_5 = tf.nn.relu(conv2d(conv2_dc2_4, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_dc2_5"))
      conv2_dc2_6 = conv2d(conv2_dc2_5, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_dc2_6")
      second_dadd1=tf.add(conv2_1_4_dc,conv2_dc2_6)
      decoder2_down2=tf.image.resize_bilinear(second_dadd1,[first_con_ff.get_shape().as_list()[1], first_con_ff.get_shape().as_list()[2]])        
# third decoder
      decoder_input2=tf.add(first_con_ff,tf.multiply(first_con_ff,(1-self._depth)))
      concate_2 = tf.concat(axis = 3, values = [decoder2_down2,decoder_input2])
      conv2_1_5_dc = tf.nn.relu(conv2d(concate_2, 256,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_1_5_dc"))      
      conv2_dc3_1 = tf.nn.relu(conv2d(conv2_1_5_dc, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_dc3_1"))
      conv2_dc3_2 = tf.nn.relu(conv2d(conv2_dc3_1, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_dc3_2"))
      conv2_dc3_3 = conv2d(conv2_dc3_2, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_dc3_3")
      third_dadd=tf.add(conv2_1_5_dc,conv2_dc3_3)
      conv2_1_6_dc = tf.nn.relu(conv2d(third_dadd, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_1_6_dc"))
      conv2_dc3_4 = tf.nn.relu(conv2d(conv2_1_6_dc, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_dc3_4"))
      conv2_dc3_5 = tf.nn.relu(conv2d(conv2_dc3_4, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_dc3_5"))
      conv2_dc3_6 = conv2d(conv2_dc3_5, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_dc3_6")
      third_dadd1=tf.add(conv2_1_6_dc,conv2_dc3_6)
      conv2_refine = conv2d(third_dadd1, 128,3,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_refine")    
 
      final_results=conv2_refine

    return final_results


 

  def save(self, checkpoint_dir, step):
    model_name = "coarse.model"
    model_dir = "%s_%s" % ("coarse", self.label_height)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "%s_%s" % ("coarse", self.label_height)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False

  def load_old(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "%s_%s" % ("coarse", self.label_height)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver_old.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False

  def Squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
    with tf.name_scope(layer_name) :
        squeeze = Global_Average_Pooling(input_x)

        excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_fully_connected1')
        excitation = Relu(excitation)
        excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_fully_connected2')
        excitation = Sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1,1,1,out_dim])

        scale = input_x * excitation

        return scale