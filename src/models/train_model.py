import tensorflow as tf
import numpy as np
import os
import time

def read_image(filename_queue):
    reader = tf.WholeFileReader()

    key, value = reader.read(filename_queue)
    original = tf.image.decode_jpeg(value, channels=3)
    original = tf.cast(original, tf.float32)

    key, value = reader.read(filename_queue)
    blurred = tf.image.decode_jpeg(value, channels=3)
    blurred = tf.cast(blurred, tf.float32)

    return original, blurred


def input_data():
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    raw_data_path = os.path.join(project_dir, "data", "raw", "pre-blur")
    raw_data_filenames = os.listdir(raw_data_path)

    blurred_data_path = os.path.join(project_dir, "data", "processed")
    blurred_data_filenames = os.listdir(blurred_data_path)

    grouped_data = []

    for image_name in blurred_data_filenames:
        corresponding_raw = image_name.split('-blurred-')[0] + '.jpg'
        grouped_data.append(os.path.join(raw_data_path, corresponding_raw)) # Append original first then blurred
        grouped_data.append(os.path.join(blurred_data_path, image_name))


    raw_data_queue = tf.train.string_input_producer(grouped_data, shuffle=False)

    original, blurred = read_image(raw_data_queue)

    input_images, ref_images = tf.train.batch(
        [blurred, original],
        batch_size = 30,
        num_threads = 1,
        capacity = 100, # Need to change this value to something meaningful
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
        activation=tf.nn.relu #Think this should actually be linear
    )

    return conv15

def loss(pred, ref):
    square_error = tf.nn.l2_loss(tf.subtract(pred, ref))
    tf.add_to_collection('losses', square_error)
    return tf.add_n(tf.get_collection("losses"), name="Total_loss")

def train(total_loss, global_step):
    return tf.train.GradientDescentOptimizer(1e-3).minimize(total_loss, global_step=global_step)

def main(argv=None):
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        input_images, ref_images = input_data()

        predicted_output = run_network(input_images)

        total_loss = loss(predicted_output, tf.cast(ref_images, dtype= tf.float32))

        train_op = train(total_loss, global_step)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        tf.train.start_queue_runners(sess=sess)

    for step in range(5):
        start_time = time.time()
        _, loss_value = sess.run([train_op, total_loss])
        duration = time.time() - start_time
        print(duration, loss_value)


if __name__ == '__main__':
    tf.app.run()
