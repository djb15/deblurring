import tensorflow as tf
import numpy as np
import os

def read_image(filename_queue):
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    image = tf.image.decode_jpeg(value, channels=3)
    return key, image


def input_data():
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    raw_data_path = os.path.join(project_dir, "data", "raw", "pre-blur")
    raw_data_filepaths = [os.path.join(raw_data_path, f) for f in os.listdir(raw_data_path)][:2]
    raw_data_queue = tf.train.string_input_producer(raw_data_filepaths)
    filename, image = read_image(raw_data_queue)
    return filename, image


def build_network(input_layer):
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=128,
        kernel=[19, 19],
        padding='same',
        activation=tf.nn.relu
    )

    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=320,
        kernel=[1, 1],
        padding='same',
        activation=tf.nn.relu
    )

    conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=320,
        kernel=[1, 1],
        padding='same',
        activation=tf.nn.relu
    )

    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=320,
        kernel=[1, 1],
        padding='same',
        activation=tf.nn.relu
    )

    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=128,
        kernel=[1, 1],
        padding='same',
        activation=tf.nn.relu
    )

    conv6 = tf.layers.conv2d(
        inputs=conv5,
        filters=128,
        kernel=[3, 3],
        padding='same',
        activation=tf.nn.relu
    )

    conv7 = tf.layers.conv2d(
        inputs=conv6,
        filters=512,
        kernel=[1, 1],
        padding='same',
        activation=tf.nn.relu
    )

    conv8 = tf.layers.conv2d(
        inputs=conv7,
        filters=128,
        kernel=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )

    conv9 = tf.layers.conv2d(
        inputs=conv8,
        filters=128,
        kernel=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )

    conv10 = tf.layers.conv2d(
        inputs=conv9,
        filters=128,
        kernel=[3, 3],
        padding='same',
        activation=tf.nn.relu
    )

    conv11 = tf.layers.conv2d(
        inputs=conv10,
        filters=128,
        kernel=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )

    conv12 = tf.layers.conv2d(
        inputs=conv11,
        filters=128,
        kernel=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )

    conv13 = tf.layers.conv2d(
        inputs=conv12,
        filters=256,
        kernel=[1, 1],
        padding='same',
        activation=tf.nn.relu
    )

    conv14 = tf.layers.conv2d(
        inputs=conv13,
        filters=64,
        kernel=[7, 7],
        padding='same',
        activation=tf.nn.relu
    )

    conv15 = tf.layers.conv2d(
        inputs=conv14,
        filters=3,
        kernel=[7, 7],
        padding='same',
        activation=tf.nn.relu #Think this should actually be linear
    )


def main(argv=None):
    label, image = input_data()
    #image_batch = tf.train.batch([image], batch_size=30)
    #label_batch = tf.train.batch([label], batch_size=30)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

if __name__ == '__main__':
    tf.app.run()
