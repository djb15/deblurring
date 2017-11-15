import time
import os
from keras.models import Sequential
from keras.layers import Conv2D
from keras import callbacks
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import numpy as np
import random


def read_images(blurred_filename, original_filename):
    blurred_file = load_img(blurred_filename)
    blurred = img_to_array(blurred_file)

    original_file = load_img(original_filename)
    original = img_to_array(original_file)

    return blurred, original


def input_data_generator(batch_size=32, directory="pre-blur-cropped"):
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    raw_data_path = os.path.join(project_dir, "data", "raw", directory)

    blurred_data_path = os.path.join(project_dir, "data", "processed")
    blurred_data_filenames = os.listdir(blurred_data_path)

    batch_blurred = np.zeros((batch_size, 20, 20, 3))
    batch_original = np.zeros((batch_size, 14, 14, 3))
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


def create_model():
    model = Sequential()
    # keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None...)
    # TODO: get input shape right
    model.add(Conv2D(128, [9, 9], padding='same', input_shape=(20, 20, 3), activation='relu', data_format="channels_last"))
    model.add(Conv2D(320, [1, 1], padding='valid', activation='relu', data_format="channels_last"))
    model.add(Conv2D(320, [1, 1], padding='valid', activation='relu', data_format="channels_last"))
    model.add(Conv2D(320, [1, 1], padding='valid', activation='relu', data_format="channels_last"))
    model.add(Conv2D(128, [1, 1], padding='valid', activation='relu', data_format="channels_last"))
    model.add(Conv2D(128, [3, 3], padding='same', activation='relu', data_format="channels_last"))
    model.add(Conv2D(512, [1, 1], padding='valid', activation='relu', data_format="channels_last"))
    model.add(Conv2D(128, [5, 5], padding='valid', activation='relu', data_format="channels_last"))
    model.add(Conv2D(128, [5, 5], padding='same', activation='relu', data_format="channels_last"))
    model.add(Conv2D(128, [3, 3], padding='valid', activation='relu', data_format="channels_last"))
    model.add(Conv2D(128, [5, 5], padding='same', activation='relu', data_format="channels_last"))
    model.add(Conv2D(128, [5, 5], padding='same', activation='relu', data_format="channels_last"))
    model.add(Conv2D(256, [1, 1], padding='valid', activation='relu', data_format="channels_last"))
    model.add(Conv2D(64, [7, 7], padding='same', activation='relu', data_format="channels_last"))
    model.add(Conv2D(3, [7, 7], padding='same', activation=None, data_format="channels_last"))

    model.compile(
            loss='mean_squared_error',
            optimizer='adam')
    return model


def create_callbacks():
    callback_list = []
    # save at the end of each epoch
    filename = time.strftime("%Y%m%d-%H%M%S") + "-{epoch:02d}-{val_loss:.2f}.hdf5"
    save_on_epoch = callbacks.ModelCheckpoint(os.path.join("models", filename))
    callback_list.append(save_on_epoch)
    return callback_list


if __name__ == "__main__":

    # # split into train and test sets
    # train_size = int(len(dataset) * 0.67)
    # test_size = len(dataset) - train_size
    # train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    # look_back = 10
    # train_x, train_y = create_dataset(train, look_back)
    # test_x, test_y = create_dataset(test, look_back)

    # # reshape input to be [samples, time steps, features]
    # train_x = np.reshape(train_x, (train_x.shape[0], look_back, train_x.shape[2]))
    # test_x = np.reshape(test_x, (test_x.shape[0], look_back, test_x.shape[2]))

    # create and fit the LSTM network
    model = create_model()
    model.fit_generator(
        input_data_generator(),
        1,  # number of epochs
        1000,  # number of image pairs per epoch
        validation_data=None,  # TODO: this needs to be another generator for validation data
        callbacks=create_callbacks())

    # save model
    filename = time.strftime("%Y%m%d-%H%M%S")
    model.save(os.path.join("models", filename + ".hdf5"))

    # # make predictions
    # train_predict = model.predict(train_x)
    # test_predict = model.predict(test_x)

    # # invert scaling transformations
    # train_predict = invert_scaling(train_predict, scaler)
    # train_y = invert_scaling(train_y, scaler)
    # test_predict = invert_scaling(test_predict, scaler)
    # test_y = invert_scaling(test_y, scaler)
    # dataset = scaler.inverse_transform(dataset)

    # # calculate root mean squared error
    # trainScore = math.sqrt(mean_squared_error(train_y, train_predict))
    # print('Train Error: %.4f RMSE' % (trainScore))
    # testScore = math.sqrt(mean_squared_error(test_y, test_predict))
    # print('Test Error: %.4f RMSE' % (testScore))
