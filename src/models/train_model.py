import time
import os
from keras.models import Sequential
from keras.layers import Conv2D
from keras import callbacks
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import numpy as np
import random


def read_images(blurred_filename, original_filename):
    """Given an image filename pair, produce a pair of numpy arrays
    """
    blurred_file = load_img(blurred_filename)
    blurred = img_to_array(blurred_file) / 255

    original_file = load_img(original_filename)
    original = img_to_array(original_file) / 255

    return blurred, original


def input_data_generator(blurred_data_filenames, batch_size=20):
    """A generator to produce batches of input-output data pairs, given a list of filenames.
    """
    batch_blurred = np.zeros((batch_size, 20, 60, 3))
    batch_original = np.zeros((batch_size, 20, 60, 3))
    while True:
        for i in range(batch_size):
            # choose random index in features
            index = random.choice(range(len(blurred_data_filenames)))
            base_name = blurred_data_filenames[index].split('-blurred-')
            corresponding_raw = base_name[0] + '-cropped-' + base_name[1]
            original_filename = os.path.join(raw_data_path, corresponding_raw)
            blurred_filename = os.path.join(blurred_data_path, blurred_data_filenames[index])  # Append original first then blurred
            batch_blurred[i], batch_original[i] = read_images(blurred_filename, original_filename)

        yield batch_blurred, batch_original


def get_test_data():
    test_data = np.zeros((100, 20, 60, 3))

    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    test_data_path = os.path.join(project_dir, "data", "raw", "test")
    test_filenames = os.listdir(test_data_path)
    for i, filename in enumerate(test_filenames):
        img_file = load_img(os.path.join(test_data_path, filename))
        width = img_file.size[0]
        height = img_file.size[1]
        h_offset = random.randint(0, height-20)
        w_offset = random.randint(0, width-60)

        w_bound = w_offset + 60
        h_bound = h_offset + 20

        img_file = img_file.crop([w_offset, h_offset, w_bound, h_bound])
        img_array = img_to_array(img_file) / 255
        test_data[i] = img_array
    return test_data


def save_predictions(predictions):
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    prediction_dir = os.path.join(project_dir, "data", "predictions")
    time_str = time.strftime("%Y%m%d-%H%M%S")

    for i, prediction in enumerate(predictions):
        prediction = prediction * 255
        img_file = array_to_img(prediction)
        img_file.save(os.path.join(prediction_dir, time_str + str(i) + ".jpeg"))


def create_model():
    model = Sequential()
    # keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None...)
    model.add(Conv2D(128, [9, 9], padding='same', input_shape=(20, 60, 3), activation='relu', data_format="channels_last"))
    model.add(Conv2D(320, [1, 1], padding='same', activation='relu', data_format="channels_last"))
    model.add(Conv2D(320, [1, 1], padding='same', activation='relu', data_format="channels_last"))
    model.add(Conv2D(320, [1, 1], padding='same', activation='relu', data_format="channels_last"))
    model.add(Conv2D(128, [1, 1], padding='same', activation='relu', data_format="channels_last"))
    model.add(Conv2D(128, [3, 3], padding='same', activation='relu', data_format="channels_last"))
    model.add(Conv2D(512, [1, 1], padding='same', activation='relu', data_format="channels_last"))
    model.add(Conv2D(128, [5, 5], padding='same', activation='relu', data_format="channels_last"))
    model.add(Conv2D(128, [5, 5], padding='same', activation='relu', data_format="channels_last"))
    model.add(Conv2D(128, [3, 3], padding='same', activation='relu', data_format="channels_last"))
    model.add(Conv2D(128, [5, 5], padding='same', activation='relu', data_format="channels_last"))
    model.add(Conv2D(128, [5, 5], padding='same', activation='relu', data_format="channels_last"))
    model.add(Conv2D(256, [1, 1], padding='same', activation='relu', data_format="channels_last"))
    model.add(Conv2D(64, [7, 7], padding='same', activation='relu', data_format="channels_last"))
    model.add(Conv2D(3, [7, 7], padding='same', activation=None, data_format="channels_last"))

    model.compile(
            loss='mean_squared_error',
            optimizer='adam',
            metrics=['accuracy'])
    return model


def create_callbacks():
    callback_list = []
    # save at the end of each epoch
    filename = time.strftime("%Y%m%d-%H%M%S") + "-{epoch:02d}-{val_loss:.2f}.hdf5"
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    save_on_epoch = callbacks.ModelCheckpoint(os.path.join(project_dir, "models", filename))

    # enable tensorboard summary output to logs dir
    tensorboard_callback = callbacks.TensorBoard(log_dir=os.path.join(project_dir, "logs"))
    callback_list.append(save_on_epoch)
    callback_list.append(tensorboard_callback)
    return callback_list


if __name__ == "__main__":

    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    raw_data_path = os.path.join(project_dir, "data", "raw", "pre-blur-cropped")

    blurred_data_path = os.path.join(project_dir, "data", "processed")
    blurred_data_filenames = os.listdir(blurred_data_path)

    validation_split_index = int(len(blurred_data_filenames) * 0.67)

    train_filenames = blurred_data_filenames[:validation_split_index]
    val_filenames = blurred_data_filenames[validation_split_index:]

    batches_per_epoch = 3000  # number of batches per epoch
    batch_size = 20  # number of images per batch
    num_epochs = 1  # number of epochs

    # create model architecture
    model = create_model()

    # fit model using generated data
    model.fit_generator(
        input_data_generator(train_filenames, batch_size),
        batches_per_epoch,
        num_epochs,
        validation_data=input_data_generator(val_filenames, batch_size),
        validation_steps=100,
        callbacks=create_callbacks())

    # save model
    filename = time.strftime("%Y%m%d-%H%M%S_final.hdf5")
    model.save(os.path.join(project_dir, "models", filename))

    # make predictions (no recombination)
    test_data = get_test_data()
    test_predictions = model.predict(test_data)
    save_predictions(test_predictions)
    print("Saved predictions!")
