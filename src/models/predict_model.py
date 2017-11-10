import tensorflow as tf
import os
import time

sess = tf.Session()
saver = tf.train.import_meta_graph('../features-115000.meta')
saver.restore(sess,tf.train.latest_checkpoint('../'))

epochs = 100

for step in range(epochs):
    _, loss_val = sess.run("Total_loss:0")
    duration = time.time() - start_time
    print(
        "Epoch {step}/{total_steps}\nBatch Loss: {:.4f}\nTime:{:.2f}s\n---"
        .format(loss_val, duration, step=step, total_steps=epochs - 1))
