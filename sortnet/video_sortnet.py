import tensorflow as tf
import numpy as np
from tensorflow.contrib.slim.python.slim.nets import resnet_v2
from tensorflow.contrib import layers, slim
import input_data

flags = tf.app.flags

np.set_printoptions(suppress=True, precision=3)

flags.DEFINE_integer('learning_rate_nl10', 5, 'Negative log-10 of the learning rate.')
flags.DEFINE_boolean('cont', True, 'continue from saved checkpoint.')
flags.DEFINE_integer('num_frames', 5, 'Number of frames to sort.')
flags.DEFINE_integer('batch_size', 4, 'Number of videos to sort at once.')
FLAGS = flags.FLAGS
learning_rate = 10 ** -FLAGS.learning_rate_nl10

'''

pass in sorted video frames!

'''


batch_size = FLAGS.batch_size 
video_size = FLAGS.num_frames
total_size = batch_size * video_size

video_data = tf.placeholder(tf.float32, [batch_size, video_size, 224, 224, 3])

batch_video_data = tf.reshape(video_data, [total_size, 224, 224, 3])

# for i in range(2):

pre_logit, epoints = resnet_v2.resnet_v2_50(
  inputs = batch_video_data,
  num_classes = None,
  # reuse = True,
  scope='resnet_v2_50'
)

orig_vars = slim.get_variables_to_restore()

with tf.variable_scope('post_conv'):
  # pre_logit = tf.reshape(pre_logit, [total_size, 2048])
  embeddings = layers.fully_connected(pre_logit, 10)
  scores = layers.fully_connected(embeddings, 1, activation_fn=None)
  scores = tf.reshape(scores, [batch_size, video_size, 1])

post_conv_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='post_conv')

def pl_kl(scores):
  # log-sum-exp. this cancels itself out
  max_score = tf.reduce_max(scores, axis=1)
  # scores -= max_score

  potentials = tf.exp(scores)

  # PL probability that a given element is first
  normalized_potentials = potentials / tf.reduce_sum(potentials, 1, keepdims=True)
  denominators = tf.cumsum(potentials, reverse=True, axis=1)

  log_potentials = scores
  log_denominators = tf.log(denominators)

  # plackett-luce
  log_likelihood = tf.reduce_sum(log_potentials, axis=1) - tf.reduce_sum(log_denominators, axis=1)
  costs = -log_likelihood
  return costs

def batch_kendall_tau(scores):
  scores = tf.squeeze(scores, axis = -1)
  one = tf.ones([video_size * (video_size + 1) / 2])
  comp = tf.contrib.distributions.fill_triangular(one)
  D1 = tf.sign(comp - tf.transpose(comp))
  
  s1 = tf.expand_dims(scores, 2)
  s2 = tf.expand_dims(scores, 1)
  D2 = tf.sign(s2 - s1)
  return -tf.reduce_sum(D1 * D2) / batch_size / (video_size * (video_size - 1))

def accuracy(scores):
  true_i = tf.range(video_size)
  true_i = tf.expand_dims(true_i, 0)
  true_i = tf.matmul(tf.ones([batch_size, 1], dtype=tf.int32), true_i)
  _, i = tf.nn.top_k(tf.squeeze(scores), k = video_size, sorted=True)
  comparison = tf.equal(true_i, i)
  comparison = tf.cast(comparison, tf.float32)
  return tf.reduce_mean(comparison)

# normalize so that cost = 0 means uniform performance
constant_shift = tf.lgamma(video_size + 1.0)

fwd_costs = pl_kl(scores)
# rev_scores = tf.reverse(scores, axis=[2])
# rev_costs = pl_kl(rev_scores)
# costs = tf.minimum(fwd_costs, rev_costs)
cost = tf.reduce_mean(fwd_costs) - constant_shift
tf.summary.scalar('cost', cost)

tau = batch_kendall_tau(scores)
tf.summary.scalar('tau', tau)

acc = accuracy(scores)
tf.summary.scalar('accuracy', acc)

opt_conv = tf.train.AdamOptimizer(learning_rate / 10)
opt_fc = tf.train.AdamOptimizer(learning_rate)

global_step_tensor = tf.Variable(1, trainable=False, name='global_step')

train_conv = opt_conv.minimize(cost, var_list = orig_vars)
train_fc = opt_fc.minimize(cost, var_list = orig_vars, global_step = global_step_tensor)

train_op = tf.group(train_conv, train_fc)

saver = tf.train.Saver(orig_vars)

restore_model = slim.assign_from_checkpoint_fn(
  './resnet_v2_50.ckpt',
  orig_vars,
  ignore_missing_vars = True
)

experiment_id = 'retrained_sortnet_%d_f%d_b%d' % (FLAGS.learning_rate_nl10, FLAGS.num_frames, FLAGS.batch_size) 

with tf.Session() as sess:
  # print(sess.run(batch_kendall_tau(tf.constant([[1, 2, 3, 4, 5], [1, 3, 2, 4, 5]], dtype=tf.float32))))
  # print(sess.run(batch_kendall_tau(tf.constant([[5, 4, 3, 2, 1]], dtype=tf.float32))))
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter('../visual_logs/' + experiment_id, sess.graph)
  sess.run(tf.global_variables_initializer())
  ckpt = tf.train.latest_checkpoint('../models/' + experiment_id + '/')
  if ckpt:
    print('found checkpoint at %s' % ckpt)
    saver.restore(sess, ckpt)
  else:
    print('initializing model from pretrained_weights...')
    restore_model(sess)
  for _ in range(10000000):
    arr, nbs, rdn, vl = input_data.read_clip(
          '../list/all.list', 
          batch_size, 
          num_frames_per_clip=video_size,
          start_pos=0, 
          shuffle=True
    )
    # arr = arr.reshape([batch_size, 16, 224, 224, 3])
    _ = sess.run(train_op, feed_dict={video_data: arr})
    time = tf.train.global_step(sess, global_step_tensor)
    print(time)
    if time % 5 == 0:
      summary, loss, t, a = sess.run([merged, cost, tau, acc], feed_dict={video_data: arr})
      train_writer.add_summary(summary, time)
      print("E: %s Cost: %f Tau: %f Acc: %f" % (experiment_id, loss, t, a))
    if time % 200 == 0:
      saver.save(sess, '../models/' + experiment_id + '/checkpoint', global_step = time)
      
  

'''
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.imshow(arr[0, 0, :, :, :])
plt.savefig('frame.png')
# print(arr.shape)
'''


