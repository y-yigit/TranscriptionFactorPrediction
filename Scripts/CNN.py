#!usr/bin/env python3

import tensorflow
from tensorflow.keras.layers import Conv1D, LSTM, Dense, MaxPooling1D, Flatten, Dropout, BatchNormalization, Embedding, Bidirectional
from tensorflow.keras.models import Sequential

"""
CCN.py

__author__ = "Yaprak Yigit"
__version__ = "0.1"

.. note:: Use numpy 1.21 for compatibility with tensorflow
"""

class CNN():
    """ Convolutional neural network for predicting motifs
    .. note:: This model is not final, layers and parameters will change
    """
    def __init__(self, input_shape, model_name):
        self.model = None
        self.input_shape = input_shape
        self.set_model(input_shape, model_name)

    def set_model(self, input_shape, model_name):
        """ Instantiates a CNN model

        The models in this method originate from the models module, this module is not imported
        because it adds additional layers. The values for the dense layers were changed

        """

        model = Sequential(name=model_name)

        if model_name == "cnn_lstm_4":
            model.add(Conv1D(filters=8, kernel_size=12, strides=1, activation='relu', input_shape=input_shape, name='conv1d_1'))
            model.add(Conv1D(filters=8, kernel_size=9, strides=1, activation='relu', input_shape=input_shape, name='conv1d_2'))
            model.add(BatchNormalization(name='batchnorm_1'))
            model.add(MaxPooling1D(pool_size=2, name='max_pooling'))
            model.add(Dropout(0.3, name='dropout_1'))
            model.add(Conv1D(filters=16, kernel_size=6, strides=1, activation='relu', input_shape=input_shape, name='conv1d_3'))
            model.add(Conv1D(filters=16, kernel_size=4, strides=1, activation='relu', input_shape=input_shape, name='conv1d_4'))
            model.add(BatchNormalization(name='batchnorm_2'))
            model.add(Dropout(0.3, name='dropout_2'))
            # model.add(MaxPooling1D(pool_size=2, name='max_pooling_2'))
            model.add(LSTM(units=16, return_sequences=True, recurrent_dropout=0.3, name='lstm'))
            model.add(Flatten(name='flatten'))
            self.model = model

        elif model_name == 'cnn_lstm_small_7_1_1_5_new':
            model.add(Conv1D(filters=8, kernel_size=8, activation='relu', input_shape=input_shape, name='conv1d'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Conv1D(filters=16, kernel_size=4, activation='relu', name='conv1d_2'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(LSTM(units=16, return_sequences=True, name='lstm'))
            model.add(Flatten())
            model.add(Dense(16, activation='relu', name='dense'))
            model.add(Dense(1, activation='softmax', name='prediction'))
            self.model = model

        elif model_name == 'cnn_lstm_small_7_1_1_4_new':
            model.add(Conv1D(filters=8, kernel_size=8, activation='relu', input_shape=input_shape))
            model.add(Dropout(0.3))
            model.add(Conv1D(filters=16, kernel_size=4, activation='relu'))
            model.add(Dropout(0.3))
            model.add(LSTM(units=16, return_sequences=True))
            model.add(MaxPooling1D(pool_size=3))
            model.add(Flatten())
            model.add(Dense(16, activation='relu'))
            model.add(Dense(1, activation='softmax', name='prediction'))
            self.model = model

        elif model_name == 'cnn_blstm_2':
            model.add(Conv1D(filters=16, kernel_size=9, strides=1, activation='relu', input_shape=input_shape,
                             name='conv1d_1'))
            model.add(Conv1D(filters=16, kernel_size=6, strides=1, activation='relu', input_shape=input_shape,
                             name='conv1d_2'))
            model.add(BatchNormalization(name='batchnorm_1'))
            model.add(MaxPooling1D(pool_size=2, name='max_pooling_1'))
            model.add(Dropout(0.3, name='dropout_1'))
            model.add(Conv1D(filters=32, kernel_size=3, strides=1, activation='relu', input_shape=input_shape,
                             name='conv1d_3'))
            model.add(Conv1D(filters=32, kernel_size=3, strides=1, activation='relu', input_shape=input_shape,
                             name='conv1d_4'))
            model.add(BatchNormalization(name='batchnorm_2'))
            model.add(MaxPooling1D(pool_size=2, name='max_pooling_2'))
            model.add(Dropout(0.3, name='dropout_2'))
            model.add(Bidirectional(LSTM(units=32, return_sequences=True, recurrent_dropout=0.3, name='blstm_1')))
            model.add(Bidirectional(LSTM(units=32, return_sequences=True, recurrent_dropout=0.3, name='blstm_2')))
            model.add(Flatten(name='flatten'))
            model.add(Dense(16, activation='relu', name='dense_0'))
            model.add(BatchNormalization(name='batchnorm_3'))
            self.model = model

    def divide_labels(self, data: object, labels: object, test_set_fraction: float):
        """ Divides the input data and labels into test and training datasets based on a fraction

        :param data: A tensor object from the module tensorflow
        :type data: object
        :param labels: A numpy array containing zeros and ones
        :type labels: object
        :param test_set_fraction: A fraction below 1
        :type test_set_fraction: float

        .. note:: If values for the dense layer are set incorrectly it will return the error:
                  "ValueError: `logits` and `labels` must have the same shape".
        """
        train_size = int(len(data) * test_set_fraction)
        self.train_data = data[:train_size]
        self.train_labels = labels[:train_size]

        self.test_data = data[train_size:]
        self.test_labels = labels[train_size:]

    def compile_model(self):
        """ Compile the model
        """
        self.model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['binary_accuracy'])

    def train_model(self):
        """ Train and test the model

        Results epochs:
            1 - 187ms/step - loss: 0.6527 - binary_accuracy: 0.9237 - val_loss: 0.1032 - val_binary_accuracy: 1.0000
            10 - 186ms/step - loss: 0.3004 - binary_accuracy: 0.9296 - val_loss: 0.1373 - val_binary_accuracy: 1.0000
        """
        self.model.fit(self.train_data, self.train_labels, epochs=10, batch_size=32, validation_data=(self.test_data,
                                                                                                      self.test_labels))
    def report_summary(self):
        """ Prints a summary of the model"""
        self.model.summary()
