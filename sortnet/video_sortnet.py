import tensorflow as tf
import numpy as np
from tensorflow.contrib.slim.python.slim.nets import resnet_v2
from tensorflow.contrib import layers, slim
import input_data

flags = tf.app.flags

np.set_printoptions(suppress=True, precision=3)

flags.DEFINE_integer('learning_rate_nl10', 4, 'Negative log-10 of the learning rate.')
flags.DEFINE_boolean('cont', True, 'continue from saved checkpoint.')
FLAGS = flags.FLAGS
learning_rate = 10 ** -FLAGS.learning_rate_nl10

'''

pass in sorted video frames!

'''


batch_size = 4 
video_size = 5
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

# pre_logit = tf.reshape(pre_logit, [total_size, 2048])
embeddings = layers.fully_connected(pre_logit, 10)
scores = layers.fully_connected(embeddings, 1, activation_fn=None)
scores = tf.reshape(scores, [batch_size, video_size, 1])

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
  one = tf.ones([video_size * (video_size + 1) / 2])
  comp = tf.contrib.distributions.fill_triangular(one)
  D1 = tf.sign(comp - tf.transpose(comp))
  
  s1 = tf.expand_dims(scores, 2)
  s2 = tf.expand_dims(scores, 1)
  D2 = tf.sign(s2 - s1)
  return -tf.reduce_sum(D1 * D2 / (video_size * (video_size - 1))) / batch_size

def accuracy(scores):
  true_i = tf.range(video_size)
  true_i = tf.expand_dims(true_i, 0)
  true_i = tf.matmul(tf.ones([batch_size, 1], dtype=tf.int32), true_i)
  _, i = tf.nn.top_k(tf.squeeze(scores), k = video_size, sorted=True)
  comparison = tf.equal(true_i, i)
  comparison = tf.cast(comparison, tf.float32)
  return tf.reduce_mean(comparison)
  

fwd_costs = pl_kl(scores)
# rev_scores = tf.reverse(scores, axis=[2])
# rev_costs = pl_kl(rev_scores)
# costs = tf.minimum(fwd_costs, rev_costs)
cost = tf.reduce_sum(fwd_costs)
tf.summary.scalar('cost', cost)

tau = batch_kendall_tau(scores)
tf.summary.scalar('tau', tau)

acc = accuracy(scores)
tf.summary.scalar('accuracy', acc)

opt = tf.train.AdamOptimizer(learning_rate)
train_op = opt.minimize(cost)

print(orig_vars)
saver = tf.train.Saver(orig_vars)

restore_model = slim.assign_from_checkpoint_fn(
  './resnet_v2_50.ckpt',
  orig_vars,
  ignore_missing_vars = True
)

with tf.Session() as sess:
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter('../visual_logs/pretrained_sortnet_%d_train_3' % FLAGS.learning_rate_nl10, sess.graph)
  sess.run(tf.global_variables_initializer())
  if not FLAGS.cont:
    print('restoring model...')
    restore_model(sess)
  else:
    saver.restore(sess, '../models/pretrained_sortnet_model_5-16000')
  for t in range(16001, 10000000):
    arr, nbs, rdn, vl = input_data.read_clip(
          '../list/all.list', 
          batch_size, 
          num_frames_per_clip=video_size,
          start_pos=0, 
          shuffle=True
    )
    # arr = arr.reshape([batch_size, 16, 224, 224, 3])
    _ = sess.run(train_op, feed_dict={video_data: arr})
    print(t)
    if t % 5 == 0:
      summary, loss, T, a = sess.run([merged, cost, tau, acc], feed_dict={video_data: arr})
      train_writer.add_summary(summary, t)
      print("Cost: %f Tau: %f Acc: %f" % (loss, T, a))
    if t % 200 == 0:
      saver.save(sess, '../models/pretrain/pretrained_sortnet_model_%d' % FLAGS.learning_rate_nl10, global_step = t)
      
  

'''
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.imshow(arr[0, 0, :, :, :])
plt.savefig('frame.png')
# print(arr.shape)
'''


