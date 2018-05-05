import tensorflow as tf
import numpy as np
from tensorflow.contrib.slim.python.slim.nets import resnet_v2
from tensorflow.contrib import layers
import input_data

np.set_printoptions(suppress=True, precision=3)

'''

pass in sorted video frames!

'''


batch_size = 2 
video_size = 5
total_size = batch_size * video_size

video_data = tf.placeholder(tf.float32, [batch_size, video_size, 224, 224, 3])

batch_video_data = tf.reshape(video_data, [total_size, 224, 224, 3])

# for i in range(2):

pre_logit, epoints = resnet_v2.resnet_v2_50(
  inputs = batch_video_data,
  num_classes = None,
  # reuse = True,
  scope = 'resnet'
)

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

fwd_costs = pl_kl(scores)
rev_scores = tf.reverse(scores, axis=[2])
rev_costs = pl_kl(rev_scores)
costs = tf.minimum(fwd_costs, rev_costs)
cost = tf.reduce_sum(costs)

opt = tf.train.AdamOptimizer(1e-5)
train_op = opt.minimize(cost)
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for t in range(1000):
    arr, nbs, rdn, vl = input_data.read_clip(
          '../list/train.list', 
          batch_size, 
          num_frames_per_clip=video_size,
          start_pos=0, 
          shuffle=True
    )
    # arr = arr.reshape([batch_size, 16, 224, 224, 3])
    ll, _ = sess.run([cost, train_op], feed_dict={video_data: arr})
    print(t, ll)

  

'''
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.imshow(arr[0, 0, :, :, :])
plt.savefig('frame.png')
# print(arr.shape)
'''


