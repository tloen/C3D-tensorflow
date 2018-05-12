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
flags.DEFINE_integer('percent_train', 16, 'Percentage of data the classifer will be trained on.')
flags.DEFINE_integer('percent_dev', 4, 'Percentage of data in the dev set.')
flags.DEFINE_boolean('big_embeddings', False, 'Whether to use size-1024 embeddings.')
flags.DEFINE_boolean('uniform_sampling', False, 'Whether to sample uniformly from the video.')
flags.DEFINE_float('lambd', 1, 'Adjust importance of sortedness.')

FLAGS = flags.FLAGS

num_labeled = int(FLAGS.batch_size / 2)
num_unlabeled = FLAGS.batch_size - num_labeled

learning_rate = 10 ** -FLAGS.learning_rate_nl10

'''

pass in sorted video frames!

'''


batch_size = FLAGS.batch_size 
video_size = FLAGS.num_frames
total_size = batch_size * video_size

video_data = tf.placeholder(tf.float32, [batch_size, video_size, 224, 224, 3])
labels = tf.placeholder(tf.int64, [num_labeled])

batch_video_data = tf.reshape(video_data, [total_size, 224, 224, 3])

# for i in range(2):

pre_logit, epoints = resnet_v2.resnet_v2_50(
  inputs = batch_video_data,
  num_classes = None,
  # reuse = True,
  scope='resnet_v2_50'
)

orig_vars = slim.get_variables_to_restore()

embed_dim = 1024 if FLAGS.big_embeddings else 10

with tf.variable_scope('post_conv'):
  # pre_logit = tf.reshape(pre_logit, [total_size, 2048])
  embeddings = layers.fully_connected(pre_logit, embed_dim, activation_fn=None)
  embed_activations = tf.nn.relu(embeddings)
  
  # sortnet
  scores = layers.fully_connected(embed_activations, 1, activation_fn=None)
  scores = tf.reshape(scores, [batch_size, video_size, 1])

  # classifier
  ea = tf.reshape(embed_activations, [batch_size, video_size, embed_dim])
  ea = ea[:num_labeled, :, :]
  pool_embed = tf.reduce_mean(ea, axis=1)
  fc1 = layers.fully_connected(pool_embed, 4096)
  fc2 = layers.fully_connected(pool_embed, 4096)
  logits = layers.fully_connected(pool_embed, 101, activation_fn=None)
  cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels, logits = logits)


post_conv_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='post_conv')

def pl_kl(scores):
  # log-sum-exp. this cancels itself out
  max_score = tf.reduce_max(scores, axis=1, keepdims=True)
  scores -= max_score

  potentials = tf.exp(scores)

  # PL probability that a given element is first
  normalized_potentials = potentials / tf.reduce_sum(potentials, 1, keepdims=True)
  denominators = tf.cumsum(potentials, reverse=True, axis=1)

  log_potentials = scores
  log_denominators = tf.log(denominators + 1e-20)

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
  return tf.reduce_sum(D1 * D2) / batch_size / (video_size * (video_size - 1))

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
cost_scalar = tf.summary.scalar('cost', cost)

tau = batch_kendall_tau(scores)
tau_scalar = tf.summary.scalar('tau', tau)

acc = accuracy(scores)
acc_scalar = tf.summary.scalar('accuracy', acc)

merged_sortnet = tf.summary.merge([cost_scalar, tau_scalar, acc_scalar])

classifier_cost = tf.reduce_mean(cross_entropy_loss)
classifier_cost_scalar = tf.summary.scalar('classifier_cost', classifier_cost)

classifier_right = tf.equal(tf.argmax(logits, 1), labels)
classifier_acc = tf.reduce_mean(tf.cast(classifier_right, tf.float32))
classifier_acc_scalar = tf.summary.scalar('classifier_acc', classifier_acc)

merged_classifier = tf.summary.merge([classifier_cost_scalar, classifier_acc_scalar])

overall_cost = classifier_cost + FLAGS.lambd * cost
overall_cost_scalar = tf.summary.scalar('overall_cost', overall_cost)
merged = tf.summary.merge([merged_sortnet, merged_classifier, overall_cost_scalar])

opt_conv = tf.train.AdamOptimizer(learning_rate / 10)
opt_fc = tf.train.AdamOptimizer(learning_rate)

global_step_tensor = tf.Variable(1, trainable=False, name='global_step')
saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

train_conv = opt_conv.minimize(overall_cost, var_list = orig_vars)
train_fc = opt_conv.minimize(overall_cost, var_list=post_conv_vars, global_step = global_step_tensor)
train_op = tf.group(train_conv, train_fc)

restore_model = slim.assign_from_checkpoint_fn(
  './resnet_v2_50.ckpt',
  orig_vars,
  ignore_missing_vars = True
)

percent_test = 100 - FLAGS.percent_train - FLAGS.percent_dev

split_id = '%d_%d_%d' % (FLAGS.percent_train, FLAGS.percent_dev, percent_test)

experiment_id = 'sc%d_l%d_f%d_b%d_d%d_' % (FLAGS.lambd * 100, FLAGS.learning_rate_nl10, FLAGS.num_frames, FLAGS.batch_size, FLAGS.percent_dev) 

if FLAGS.uniform_sampling:
  experiment_id += '_us'

if FLAGS.big_embeddings:
  experiment_id += '_big'

with tf.Session() as sess:
  # print(sess.run(batch_kendall_tau(tf.constant([[1, 2, 3, 4, 5], [1, 3, 2, 4, 5]], dtype=tf.float32))))
  # print(sess.run(batch_kendall_tau(tf.constant([[5, 4, 3, 2, 1]], dtype=tf.float32))))
  train_writer = tf.summary.FileWriter('../visual_logs/%s' % experiment_id, sess.graph)
  dev_writer = tf.summary.FileWriter('../visual_logs/%s_dev' % experiment_id, sess.graph)
  sess.run(tf.global_variables_initializer())
  ckpt = tf.train.latest_checkpoint('../models/' + experiment_id + '/')
  if ckpt:
    print('found checkpoint at %s' % ckpt)
    saver.restore(sess, ckpt)
  else:
    print('initializing model from pretrained weights...')
    restore_model(sess)
  for _ in range(10000000):
    train_arr, _, _, _, lab = input_data.read_clip(
	  '../list/s5_train_%s.list' % split_id
	    if not FLAGS.uniform_sampling
	    else '../list/s_train_%s.list' % split_id, 
	  num_labeled,
	  num_frames_per_clip=video_size,
	  start_pos=0, 
	  shuffle=True,
	  uniform_sampling=FLAGS.uniform_sampling
    )
    test_arr, _, _, _, _ = input_data.read_clip(
	  '../list/s5_test_%s.list' % split_id
	    if not FLAGS.uniform_sampling
	    else '../list/s_test_%s.list' % split_id, 
	  num_unlabeled,
	  num_frames_per_clip=video_size,
	  start_pos=0, 
	  shuffle=True,
	  uniform_sampling=FLAGS.uniform_sampling
    )
    arr = np.concatenate((train_arr, test_arr), axis = 0)
    sess.run(train_op, feed_dict={video_data:arr, labels:lab})
    time = tf.train.global_step(sess, global_step_tensor)
    print(time)
    if time % 10 == 0:
      summary, loss, t, a, closs, cacc, oc = sess.run([merged, cost, tau, acc, classifier_cost, classifier_acc, overall_cost], feed_dict={video_data: arr, labels:lab})
      train_writer.add_summary(summary, time)
      print("T | E: %s S. Cost: %f Tau: %f S. Acc: %f C. cost: %f C. Acc: %f LOSS: %f" % (experiment_id, loss, t, a, closs, cacc, oc))
      arr, nbs, rdn, vl, lab = input_data.read_clip(
	    '../list/s5_dev_%s.list' % split_id
              if not FLAGS.uniform_sampling
              else '../list/s_dev_%s.list' % split_id, 
	    batch_size, 
	    num_frames_per_clip=video_size,
	    start_pos=0, 
	    shuffle=True,
            uniform_sampling=FLAGS.uniform_sampling
      )
      summary, loss, t, a, closs, cacc, oc = sess.run([merged, cost, tau, acc, classifier_cost, classifier_acc, overall_cost], feed_dict={video_data: arr, labels:lab[:num_labeled]})
      print("V | E: %s S. Cost: %f Tau: %f S. Acc: %f C. cost: %f C. Acc: %f LOSS: %f" % (experiment_id, loss, t, a, closs, cacc, oc))
      dev_writer.add_summary(summary, time)
    if time % 1000 == 0:
      saver.save(sess, '../models/' + experiment_id + '/checkpoint', global_step = time)
