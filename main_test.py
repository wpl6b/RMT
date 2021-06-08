from model_test import T_CNN
from utils import *
import numpy as np
import tensorflow as tf

import pprint
import os
import PIL
flags = tf.app.flags
flags.DEFINE_integer("epoch", 120, "Number of epoch [120]")
flags.DEFINE_integer("batch_size", 1, "The size of batch images [128]")
flags.DEFINE_integer("image_height", 128, "The size of image to use [230]")
flags.DEFINE_integer("image_width", 128, "The size of image to use [310]")
flags.DEFINE_integer("label_height", 128, "The size of label to produce [230]")
flags.DEFINE_integer("label_width", 128, "The size of label to produce [310]")
flags.DEFINE_float("learning_rate", 0.0001, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_integer("c_depth_dim", 1, "Dimension of depth. [1]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("sample_dir", "sample", "Name of sample directory [sample]")
flags.DEFINE_string("test_data_dir", "test", "Name of sample directory [test]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [True]")
FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)
  filenames = os.listdir('input_90') ###### input image dataset
  data_dir = os.path.join(os.getcwd(), 'input_90')
  data = sorted(glob.glob(os.path.join(data_dir, "*.png")))
  test_data_list = data + sorted(glob.glob(os.path.join(data_dir, "*.jpg")))+sorted(glob.glob(os.path.join(data_dir, "*.bmp")))+sorted(glob.glob(os.path.join(data_dir, "*.jpeg")))

  filenames1 = os.listdir('gdcp_90') ###### input transmission dataset
  data_dir1 = os.path.join(os.getcwd(), 'gdcp_90')
  data1 = sorted(glob.glob(os.path.join(data_dir1, "*.png")))
  test_data_list1 = data1 + sorted(glob.glob(os.path.join(data_dir1, "*.jpg")))+sorted(glob.glob(os.path.join(data_dir1, "*.bmp")))+sorted(glob.glob(os.path.join(data_dir1, "*.jpeg")))

  for ide in range(0,len(test_data_list)):
    image_test1 =  get_image(test_data_list[ide],is_grayscale=False)
    shape = image_test1.shape
    RGB=Image.fromarray(np.uint8(image_test1*255))
    RGB1=RGB.resize(((shape[1]//8-0)*8,(shape[0]//8-0)*8))
    image_test = np.asarray(np.float32(RGB1)/255)
    depth_test1 =  get_image(test_data_list1[ide],is_grayscale=False)
    Depth=Image.fromarray(np.uint8(depth_test1*255))
    Depth1=Depth.resize(((shape[1]//8-0)*8,(shape[0]//8-0)*8))
    depth_test = np.asarray(np.float32(Depth1)/255)

    shape = image_test.shape
    tf.reset_default_graph()
    with tf.Session() as sess:
      # with tf.device('/cpu:0'):
      # 
        srcnn = T_CNN(sess, 
                  image_height=shape[0],
                  image_width=shape[1],  
                  label_height=FLAGS.label_height, 
                  label_width=FLAGS.label_width, 
                  batch_size=FLAGS.batch_size,
                  c_dim=FLAGS.c_dim, 
                  c_depth_dim=FLAGS.c_depth_dim,
                  checkpoint_dir=FLAGS.checkpoint_dir,
                  sample_dir=FLAGS.sample_dir,
                  test_image_name = test_data_list[ide],
                  test_depth_name = test_data_list1[ide],
                  id = ide
                  )

        srcnn.train(FLAGS)
        sess.close()
    tf.get_default_graph().finalize()
if __name__ == '__main__':
  tf.app.run()
