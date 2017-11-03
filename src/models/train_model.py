import tensorflow as tf
import numpy as np
import os
import time


def read_image(filename_queue):

    value = tf.read_file(filename_queue[0])
    original = tf.image.decode_jpeg(value, channels=3)
    original = tf.cast(original, tf.float32)

    value = tf.read_file(filename_queue[1])
    blurred = tf.image.decode_jpeg(value, channels=3)
    blurred = tf.cast(blurred, tf.float32)

    return original, blurred


def input_data(batch_size):
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    raw_data_path = os.path.join(project_dir, "data", "raw", "pre-blur")
    raw_data_filenames = os.listdir(raw_data_path)

    blurred_data_path = os.path.join(project_dir, "data", "processed")
    blurred_data_filenames = os.listdir(blurred_data_path)

    raw_data = []
    blurred_data = []

    for image_name in blurred_data_filenames:
        corresponding_raw = image_name.split('-blurred-')[0] + '.jpg'
        raw_data.append(os.path.join(raw_data_path, corresponding_raw))
        blurred_data.append(os.path.join(blurred_data_path, image_name))  # Append original first then blurred

    raw_data_queue = tf.train.slice_input_producer([raw_data, blurred_data])

    original, blurred = read_image(raw_data_queue)

    input_images, ref_images = tf.train.batch(
        [blurred, original],
        batch_size=batch_size,
        num_threads=1,
        capacity=40,  # The prefetch buffer for this batch train
        dynamic_pad=True
    )

    return input_images, ref_images


def run_network(input_layer):
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=128,
        kernel_size=[19, 19],
        padding='same',
        activation=tf.nn.relu
    )

    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=320,
        kernel_size=[1, 1],
        padding='same',
        activation=tf.nn.relu
    )

    conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=320,
        kernel_size=[1, 1],
        padding='same',
        activation=tf.nn.relu
    )

    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=320,
        kernel_size=[1, 1],
        padding='same',
        activation=tf.nn.relu
    )

    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=128,
        kernel_size=[1, 1],
        padding='same',
        activation=tf.nn.relu
    )

    conv6 = tf.layers.conv2d(
        inputs=conv5,
        filters=128,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu
    )

    conv7 = tf.layers.conv2d(
        inputs=conv6,
        filters=512,
        kernel_size=[1, 1],
        padding='same',
        activation=tf.nn.relu
    )

    conv8 = tf.layers.conv2d(
        inputs=conv7,
        filters=128,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )

    conv9 = tf.layers.conv2d(
        inputs=conv8,
        filters=128,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )

    conv10 = tf.layers.conv2d(
        inputs=conv9,
        filters=128,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu
    )

    conv11 = tf.layers.conv2d(
        inputs=conv10,
        filters=128,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )

    conv12 = tf.layers.conv2d(
        inputs=conv11,
        filters=128,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )

    conv13 = tf.layers.conv2d(
        inputs=conv12,
        filters=256,
        kernel_size=[1, 1],
        padding='same',
        activation=tf.nn.relu
    )

    conv14 = tf.layers.conv2d(
        inputs=conv13,
        filters=64,
        kernel_size=[7, 7],
        padding='same',
        activation=tf.nn.relu
    )

    conv15 = tf.layers.conv2d(
        inputs=conv14,
        filters=3,
        kernel_size=[7, 7],
        padding='same',
        activation=None
    )

    return conv15


def loss(pred, ref):
    mse = tf.losses.mean_squared_error(ref, pred)
    tf.add_to_collection('losses', mse)
    return tf.add_n(tf.get_collection("losses"), name="Total_loss")


def train(total_loss, global_step, learning_rate):
    decay_steps = 50
    decay_rate = 0.96
    decayed_learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate)
    return tf.train.GradientDescentOptimizer(
        decayed_learning_rate).minimize(total_loss, global_step=global_step)


def main(argv=None):
    learning_rate = 1e-6  # higher causes NaN issues
    epochs = 5
    batch_size = 20  # higher causes OOM issues on 4GB GPU

    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        input_images, ref_images = input_data(batch_size)

        predicted_output = run_network(input_images)

        loss_op = loss(predicted_output, tf.cast(ref_images, dtype=tf.float32))

        train_op = train(loss_op, global_step, learning_rate)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        tf.train.start_queue_runners(sess=sess)

    for step in range(epochs):
        start_time = time.time()
        _, loss_val = sess.run([train_op, loss_op])
        duration = time.time() - start_time
        print(
            "Epoch {step}/{total_steps}\nBatch Loss: {:.4f}\nTime:{:.2f}s\n---"
            .format(loss_val, duration, step=step, total_steps=epochs))

if __name__ == '__main__':
    tf.app.run()
