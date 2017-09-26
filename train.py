import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim

from dataset import BatchGenerator
from model import *
from yasuo import Yasuo

flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_integer('train_iter', 450, 'Total training iter')
flags.DEFINE_integer('step', 500, 'Save after ... iteration')

#mnist = get_mnist()
#
#test_im = np.array([im.reshape((28,28,1)) for im in mnist.test.images])

hasaki = Yasuo('Data/pair.txt')

gen = BatchGenerator(hasaki.Train.simA, hasaki.Train.simB, hasaki.Train.label)

# len(hasaki.Train.simA)) = len(hasaki.Train.simB)) = len(hasaki.Train.label))


left = tf.placeholder(tf.float32, [None, 256, 256, 3], name='left')
right = tf.placeholder(tf.float32, [None, 256, 256, 3], name='right')

with tf.name_scope("similarity"):
	label = tf.placeholder(tf.int32, [None, 1], name='label') # 1 if same, 0 if different
	label = tf.to_float(label)
    
margin = 0.2

left_output = network(left, reuse=False)

right_output = network(right, reuse=True)

loss = contrastive_loss(left_output, right_output, label, margin)

global_step = tf.Variable(0, trainable=False)

train_step = tf.train.MomentumOptimizer(0.01, 0.99, use_nesterov=True).minimize(loss, global_step=global_step)


saver = tf.train.Saver()
with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())

	#setup tensorboard	
   tf.summary.scalar('step', global_step)
   tf.summary.scalar('loss', loss)
   for var in tf.trainable_variables():
      tf.summary.histogram(var.op.name, var)
   merged = tf.summary.merge_all()
   writer = tf.summary.FileWriter('train.log', sess.graph)
   
   coord = tf.train.Coordinator()
   threads = tf.train.start_queue_runners(coord=coord)
	#train iter
   for i in range(FLAGS.train_iter):

      b_l, b_r, b_sim = gen.next_batch(FLAGS.batch_size)
      _, l, summary_str = sess.run([train_step, loss, merged], feed_dict={left:b_l, right:b_r, label: b_sim})
		
        
      coord.request_stop()
      coord.join(threads)
        
      writer.add_summary(summary_str, i)
      print ("\r#%d - Loss"%i, l)

   saver.save(sess, "model/model.ckpt")




