#!usr/bin/env python3

from tensorflow.keras.layers import Conv1D, LSTM, Dense, MaxPooling1D, Flatten, Dropout, BatchNormalization
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
    def __init__(self, input_shape):
        self.model = None
        self.set_model(input_shape)

    def set_model(self, input_shape):
        """ Instantiates CNN model "cnn_lstm_4" from the models package
        """
        model = Sequential(name='model_cnn_lstm')
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

    def divide_labels(self, data: object, labels: object, test_set_fraction: float):
        """ Divides the input data and labels into test and training datasets based on a fraction

        :param data: A tensor object from the module tensorflow
        :type data: object
        :param labels: A numpy array containing zeros and ones
        :type labels: object
        :param test_set_fraction: A fraction below 1
        :type test_set_fraction: float

        .. note:: If values for the dense layer are set incorrectly it will return the error:
                  "ValueError: `logits` and `labels` must have the same shape". This error is not a

        .. todo:: The true positives and true negatives in the input data are not shuffled, this method should do that
        """
        train_split = 0.8  # 80% for training, 20% for testing
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