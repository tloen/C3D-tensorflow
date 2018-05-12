import tensorflow as tf
import numpy as np
from tensorflow.contrib.slim.python.slim.nets import resnet_v2
from tensorflow.contrib import layers, slim
import input_data

flags = tf.app.flags

np.set_printoptions(suppress=True, precision=3)
flags.DEFINE_string('experiment_id', 'retrained_sortnet_l4_f8_b10_d4', 'name of the saved model to retrieve.')
FLAGS = flags.FLAGS

experiment_id = FLAGS.experiment_id

video_size = 16

video_data = tf.placeholder(tf.float32, [None, 224, 224, 3])
batch_video_data = tf.reshape(video_data, [-1, 224, 224, 3])

# for i in range(2):

pre_logit, epoints = resnet_v2.resnet_v2_50(
  inputs = batch_video_data,
  num_classes = None,
  # reuse = True,
  scope='resnet_v2_50'
)

with tf.variable_scope('post_conv'):
  # pre_logit = tf.reshape(pre_logit, [total_size, 2048])
  embeddings = layers.fully_connected(pre_logit, 10, activation_fn=None)
  activations = tf.nn.relu(embeddings)
  scores = layers.fully_connected(activations, 1, activation_fn=None)
  scores = tf.reshape(scores, [-1, 1])

embeddings = tf.squeeze(embeddings)
post_conv_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='post_conv')
saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
ckpt = tf.train.latest_checkpoint('/sailhome/ejwang/projects/C3D-tensorflow/models/' + experiment_id + '/')
if ckpt:
  print('found checkpoint at %s' % ckpt)
  saver.restore(sess, ckpt)
else:
  print('checkpoint NOT FOUND')
  exit()

def get_embeddings(arr):
  return sess.run(embeddings, {video_data: arr}) 

'''
with open('../list/all.list', 'r') as index_file:
  for line in index_file:
    filename, l = line.split(' ')
    arr, img_names = input_data.read_clip_full(filename) 
    print(arr)
    print(arr.shape)
    # exit()  
    emb = sess.run(embeddings, {video_data: arr})   
    print(emb)
'''












