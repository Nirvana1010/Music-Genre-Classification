"""
This file is the training part of LSTM model with input file is the json file generated from
LSTM_Data_Preprocessing file.

The python file is referenced and modified from
https://github.com/rajatkeshri/Music-Genre-Prediction-Using-RNN-LSTM

This file requires packages:
tensorflow 2.3.1
sklearn 1.0.2
numpy 1.18.5
"""
import os
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

data_path = ".\Data\data_json"


def load_data(data_path):
    print("Data loading\n")
    with open(data_path, "r") as fp:
        data = json.load(fp)

    x = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("Loaded Data")

    return x, y


def prepare_datasets(test_size, val_size):
    # load the data
    x, y = load_data(data_path)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=val_size)

    return x_train, x_val, y_train, y_val


def build_model(input_shape):
    model = tf.keras.Sequential()
    print('inputshape', input_shape)
    model.add(tf.keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(tf.keras.layers.LSTM(64))

    model.add(tf.keras.layers.Dense(64, activation="relu"))
    # model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    return model


def plot_history(history):
    print(history.history.keys())
    #  "Accuracy"
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


if __name__ == "__main__":
    x_train, x_val, y_train, y_val = prepare_datasets(0.3, 0.3)

    input_shape = (x_train.shape[1], x_train.shape[2])
    model = build_model(input_shape)

    # compile model
    optimiser = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # train model
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=32, epochs=50)

    # plot accuracy/error for training and validation
    plot_history(history)
