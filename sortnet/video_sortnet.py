import tensorflow as tf
import numpy as np
from tensorflow.contrib.slim.python.slim.nets import resnet_v2
from tensorflow.contrib import layers
import input_data

'''

pass in sorted video frames!

'''

batch_size = 16
video_data = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])

pre_logit, epoints = resnet_v2.resnet_v2_152(
  inputs = video_data,
  num_classes = None
)

pre_logit = tf.reshape(pre_logit, [batch_size, 2048])
embeddings = layers.fully_connected(pre_logit, 10)
scores = layers.fully_connected(embeddings, 1)

# log-sum-exp. this cancels itself out
max_score = tf.reduce_max(scores)
scores -= max_score

potentials = tf.exp(scores)
denominators = tf.cumsum(potentials, reverse=True)

log_potentials = scores
log_denominators = tf.log(denominators)

# plackett-luce
log_likelihoood = tf.reduce_sum(log_potentials) - tf.reduce_sum(log_denominators)
cost = -log_likelihood
print(embeddings)


arr, nbs, rdn, vl = read_clip('../list/train.list', 10, num_frames_per_clip=16, shuffle=True)

print(arr.shape)
