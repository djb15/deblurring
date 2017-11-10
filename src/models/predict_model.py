import tensorflow as tf
import os

sess = tf.Session()
saver = tf.train.import_meta_graph('trained-1000.meta')
saver.restore(sess,tf.train.latest_checkpoint('../'))

sess.run("predict:0")
