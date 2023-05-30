#!usr/bin/env python3

import tensorflow
from tensorflow import keras
from tensorflow.keras import regularizers, optimizers, losses
from tensorflow.keras.callbacks import ModelCheckpoint, History, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Conv1D, Masking, LSTM, Dense, MaxPooling1D, Flatten, Dropout, BatchNormalization, \
    Embedding, Bidirectional, GlobalMaxPool1D, InputLayer, GaussianNoise, AveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2, l1
import keras_tuner as kt
from keras_tuner import RandomSearch, HyperParameters

from tensorflow.keras.metrics import SensitivityAtSpecificity
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, classification_report

"""
CCN.py

__author__ = "Yaprak Yigit"
__version__ = "0.1"

.. note:: Use numpy 1.21 for compatibility with tensorflow
.. note:: DENSE LAYER DECIDES THE AMOUNT OF CLASSES
"""


class CNN():
    """ Convolutional neural network for predicting motifs
    .. note:: This model is not final, layers and parameters will change
    """

    def __init__(self, input_shape, model_name=None):
        self.model = None
        self.input_shape = input_shape
        if model_name:
            self.set_model(input_shape, model_name)

    def set_model(self, input_shape, model_name):
        """ Instantiates a CNN model
        """

        model = Sequential(name=model_name)

        if model_name == "model_gaussian":
            model.add(Masking(mask_value=0))
            model.add(GaussianNoise(0.2))
            model.add(Conv1D(filters=128, kernel_size=5, activation='relu', kernel_regularizer=regularizers.l2(1e-4),
                             bias_regularizer=regularizers.l2(1e-6), input_shape=input_shape))
            model.add(MaxPooling1D(pool_size=4, strides=2))
            model.add(Dropout(0.5))
            model.add(Conv1D(filters=256, kernel_size=7, activation='relu', kernel_regularizer=regularizers.l2(1e-4),
                             bias_regularizer=regularizers.l2(1e-6)))
            model.add(MaxPooling1D(pool_size=4, strides=2))
            model.add(Dropout(0.5))
            model.add(Flatten())
            model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4),
                            bias_regularizer=regularizers.l2(1e-6)))
            model.add(Dropout(0.5))
            model.add(Dense(2, activation='sigmoid'))
            self.model = model

        elif model_name == 'model_lstm_bi':
            model.add(Masking(mask_value=0))
            model.add(Bidirectional(LSTM(units=16, return_sequences=True, name='lstm'), input_shape=input_shape))
            model.add(Conv1D(filters=32, kernel_size=8, name='conv1d'))
            model.add(MaxPooling1D(pool_size=4, name='max_pooling'))
            model.add(Flatten(name='flatten'))
            model.add(Dropout(0.5))
            model.add(Dense(16, activation='relu', name='dense'))
            model.add(Dense(2, activation='softmax', name='prediction'))
            self.model = model

    def model_builder(self, hp):
        hp_units_lstm = hp.Int('units_lstm', min_value=16, max_value=128, step=16)
        hp_units_conv = hp.Int('units_conv', min_value=16, max_value=128, step=16)
        hp_units_dense = hp.Int('units_dense', min_value=16, max_value=128, step=16)

        hp_dense_activation = hp.Choice('dense_activation', values=['relu', 'tanh', 'sigmoid'])
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        hp_pool_size = hp.Choice('pool_size', values=[2, 3, 4])
        hp_kernel_size = hp.Choice('kernel_size', values=[3, 5, 7, 8, 9, 11])
        hp_conv_layers = hp.Int('conv_layers', min_value=1, max_value=3)
        hp_pool_type = hp.Choice('pool_type', values=['max', 'average'])

        hp_lstm_activation = hp.Choice('lstm_activation', values=['relu', 'tanh', 'sigmoid'])
        hp_dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
        hp_pool_padding = hp.Choice('pool_padding', values=['valid', 'same'])

        model = keras.Sequential()
        model.add(Masking(mask_value=0))
        model.add(
            Bidirectional(LSTM(units=hp_units_lstm, activation=hp_lstm_activation, return_sequences=True, name='lstm'),
                          input_shape=self.input_shape))
        for _ in range(hp_conv_layers):
            model.add(Conv1D(filters=hp_units_conv, kernel_size=hp_kernel_size, name='conv1d'))
        if hp_pool_type == 'max':
            model.add(MaxPooling1D(pool_size=hp_pool_size, padding=hp_pool_padding, name='max_pooling'))
        elif hp_pool_type == 'average':
            model.add(AveragePooling1D(pool_size=hp_pool_size, padding=hp_pool_padding, name='average_pooling'))
        model.add(Flatten(name='flatten'))
        model.add(Dropout(hp_dropout_rate))
        model.add(Dense(hp_units_dense, activation=hp_dense_activation, name='dense'))
        model.add(Dense(2, activation='softmax', name='prediction'))

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def create_hyper_model(self):
        tuner = kt.Hyperband(self.model_builder,
                             objective='val_accuracy',
                             max_epochs=100,
                             factor=3,
                             directory='my_dir',
                             project_name='intro_to_kt')
        stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0)

        tuner.search(self.train_data, self.train_labels, epochs=100, validation_split=0.2, callbacks=[stop_early])

        # Get the optimal hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=50)[0]

        # Build the model with the optimal hyperparameters and train it on the data for 100 epochs
        model = tuner.hypermodel.build(best_hps)

        # Build the model with the optimal hyperparameters and train it on the data for 100 epochs
        model = tuner.hypermodel.build(best_hps)
        history = model.fit(self.train_data, self.train_labels, epochs=100, validation_split=0.2)

        val_acc_per_epoch = history.history['val_accuracy']
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1

        # Re-instantiate the hypermodel and train it with the optimal number of epochs from above.
        hypermodel = tuner.hypermodel.build(best_hps)
        # Retrain the model
        res = hypermodel.fit(self.train_data, self.train_labels, epochs=best_epoch, validation_split=0.2)
        return hypermodel, history, best_epoch

    def set_input(self, train_data: object, test_data, train_labels: object, test_labels):
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
        test_set_fraction = 0.8  # 80% for training, 20% for testing
        train_size = int(len(train_data) * test_set_fraction)

        self.train_data = train_data[:train_size]
        self.train_labels = train_labels[:train_size]

        self.val_data = train_data[train_size:]
        self.val_labels = train_labels[train_size:]

        self.test_data = test_data
        self.test_labels = test_labels

    def compile_model(self):
        """ Compile the model
        """

        self.model.compile(loss='binary_crossentropy',
                           optimizer="adam",  # tensorflow.keras.optimizers.Adam(learning_rate=0.001),
                           metrics=['binary_accuracy'])  # , SensitivityAtSpecificity(0.9)])

    def train_model(self, epochs, batch_size):
        """ Train and test the model

        Results epochs:
            1 - 187ms/step - loss: 0.6527 - binary_accuracy: 0.9237 - val_loss: 0.1032 - val_binary_accuracy: 1.0000
            10 - 186ms/step - loss: 0.3004 - binary_accuracy: 0.9296 - val_loss: 0.1373 - val_binary_accuracy: 1.0000
        """
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
        early_stopping = EarlyStopping(monitor='val_loss', patience=8, mode='min', restore_best_weights=True)
        res = self.model.fit(self.train_data, self.train_labels, epochs=epochs, shuffle=False,
                             batch_size=batch_size, validation_data=(self.val_data, self.val_labels),
                             callbacks=[reduce_lr, early_stopping],
                             verbose=0)

        self.predict_labels(self.val_data, self.val_labels)

    def predict_labels(self, data, labels):

        # Make predictions on test data
        predicted_labels = self.model.predict(data)

        cm = confusion_matrix(np.argmax(labels, axis=1), np.argmax(predicted_labels, axis=1))
        TP = cm[1][1]
        FP = cm[0][1]
        FN = cm[1][0]
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * (precision * recall) / (precision + recall)

        # print(f"F1 score: {f1} \nPrecision: {precision} \nRecall: {recall} \nConfusion Matrix :\n{cm}")

    @staticmethod
    def decoder(encoded_lists):
        reverse_mapping = {0: "A", 1: "G", 2: "C", 3: "T", 4: "0"}
        motifs = []

        for encoded_array in encoded_lists:
            decoded_sequence = ''.join(reverse_mapping[np.argmax(encoded_array)] for encoded_array in encoded_array)
            motifs.append(decoded_sequence)
        return motifs

    def report_summary(self):
        """ Prints a summary of the model"""
        self.model.summary()


# !usr/bin/env python3

import tensorflow
from tensorflow import keras
from tensorflow.keras import regularizers, optimizers, losses
from tensorflow.keras.callbacks import ModelCheckpoint, History, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Conv1D, Masking, LSTM, Dense, MaxPooling1D, Flatten, Dropout, BatchNormalization, \
    Embedding, Bidirectional, GlobalMaxPool1D, InputLayer, GaussianNoise, AveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2, l1
import keras_tuner as kt
from keras_tuner import RandomSearch, HyperParameters

from tensorflow.keras.metrics import SensitivityAtSpecificity
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, classification_report

"""
CCN.py

__author__ = "Yaprak Yigit"
__version__ = "0.1"

.. note:: Use numpy 1.21 for compatibility with tensorflow
.. note:: DENSE LAYER DECIDES THE AMOUNT OF CLASSES
"""


class CNN():
    """ Convolutional neural network for predicting motifs
    .. note:: This model is not final, layers and parameters will change
    """

    def __init__(self, input_shape, model_name=None):
        self.model = None
        self.input_shape = input_shape
        if model_name:
            self.set_model(input_shape, model_name)

    def set_model(self, input_shape, model_name):
        """ Instantiates a CNN model
        """

        model = Sequential(name=model_name)

        if model_name == "model_gaussian":
            model.add(Masking(mask_value=0))
            model.add(GaussianNoise(0.2))
            model.add(Conv1D(filters=128, kernel_size=5, activation='relu', kernel_regularizer=regularizers.l2(1e-4),
                             bias_regularizer=regularizers.l2(1e-6), input_shape=input_shape))
            model.add(MaxPooling1D(pool_size=4, strides=2))
            model.add(Dropout(0.5))
            model.add(Conv1D(filters=256, kernel_size=7, activation='relu', kernel_regularizer=regularizers.l2(1e-4),
                             bias_regularizer=regularizers.l2(1e-6)))
            model.add(MaxPooling1D(pool_size=4, strides=2))
            model.add(Dropout(0.5))
            model.add(Flatten())
            model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4),
                            bias_regularizer=regularizers.l2(1e-6)))
            model.add(Dropout(0.5))
            model.add(Dense(2, activation='sigmoid'))
            self.model = model

        elif model_name == 'model_lstm_bi':
            model.add(Masking(mask_value=0))
            model.add(Bidirectional(LSTM(units=16, return_sequences=True, name='lstm'), input_shape=input_shape))
            model.add(Conv1D(filters=32, kernel_size=8, name='conv1d'))
            model.add(MaxPooling1D(pool_size=4, name='max_pooling'))
            model.add(Flatten(name='flatten'))
            model.add(Dropout(0.5))
            model.add(Dense(16, activation='relu', name='dense'))
            model.add(Dense(2, activation='softmax', name='prediction'))
            self.model = model

    def model_builder(self, hp):
        hp_units_lstm = hp.Int('units_lstm', min_value=16, max_value=128, step=16)
        hp_units_conv = hp.Int('units_conv', min_value=16, max_value=128, step=16)
        hp_units_dense = hp.Int('units_dense', min_value=16, max_value=128, step=16)

        hp_dense_activation = hp.Choice('dense_activation', values=['relu', 'tanh', 'sigmoid'])
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        hp_pool_size = hp.Choice('pool_size', values=[2, 3, 4])
        hp_kernel_size = hp.Choice('kernel_size', values=[3, 5, 7, 8, 9, 11])
        hp_conv_layers = hp.Int('conv_layers', min_value=1, max_value=3)
        hp_pool_type = hp.Choice('pool_type', values=['max', 'average'])

        hp_lstm_activation = hp.Choice('lstm_activation', values=['relu', 'tanh', 'sigmoid'])
        hp_dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
        hp_pool_padding = hp.Choice('pool_padding', values=['valid', 'same'])

        model = keras.Sequential()
        model.add(Masking(mask_value=0))
        model.add(
            Bidirectional(LSTM(units=hp_units_lstm, activation=hp_lstm_activation, return_sequences=True, name='lstm'),
                          input_shape=self.input_shape))
        for _ in range(hp_conv_layers):
            model.add(Conv1D(filters=hp_units_conv, kernel_size=hp_kernel_size, name='conv1d'))
        if hp_pool_type == 'max':
            model.add(MaxPooling1D(pool_size=hp_pool_size, padding=hp_pool_padding, name='max_pooling'))
        elif hp_pool_type == 'average':
            model.add(AveragePooling1D(pool_size=hp_pool_size, padding=hp_pool_padding, name='average_pooling'))
        model.add(Flatten(name='flatten'))
        model.add(Dropout(hp_dropout_rate))
        model.add(Dense(hp_units_dense, activation=hp_dense_activation, name='dense'))
        model.add(Dense(2, activation='softmax', name='prediction'))

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def create_hyper_model(self):
        tuner = kt.Hyperband(self.model_builder,
                             objective='val_accuracy',
                             max_epochs=100,
                             factor=3,
                             directory='my_dir',
                             project_name='intro_to_kt')
        stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0)

        tuner.search(self.train_data, self.train_labels, epochs=100, validation_split=0.2, callbacks=[stop_early])

        # Get the optimal hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=50)[0]

        # Build the model with the optimal hyperparameters and train it on the data for 100 epochs
        model = tuner.hypermodel.build(best_hps)

        # Build the model with the optimal hyperparameters and train it on the data for 100 epochs
        model = tuner.hypermodel.build(best_hps)
        history = model.fit(self.train_data, self.train_labels, epochs=100, validation_split=0.2)

        val_acc_per_epoch = history.history['val_accuracy']
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1

        # Re-instantiate the hypermodel and train it with the optimal number of epochs from above.
        hypermodel = tuner.hypermodel.build(best_hps)
        # Retrain the model
        res = hypermodel.fit(self.train_data, self.train_labels, epochs=best_epoch, validation_split=0.2)
        return hypermodel, history, best_epoch

    def set_input(self, train_data: object, test_data, train_labels: object, test_labels):
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
        test_set_fraction = 0.8  # 80% for training, 20% for testing
        train_size = int(len(train_data) * test_set_fraction)

        self.train_data = train_data[:train_size]
        self.train_labels = train_labels[:train_size]

        self.val_data = train_data[train_size:]
        self.val_labels = train_labels[train_size:]

        self.test_data = test_data
        self.test_labels = test_labels

    def compile_model(self):
        """ Compile the model
        """

        self.model.compile(loss='binary_crossentropy',
                           optimizer="adam",  # tensorflow.keras.optimizers.Adam(learning_rate=0.001),
                           metrics=['binary_accuracy'])  # , SensitivityAtSpecificity(0.9)])

    def train_model(self, epochs, batch_size):
        """ Train and test the model

        Results epochs:
            1 - 187ms/step - loss: 0.6527 - binary_accuracy: 0.9237 - val_loss: 0.1032 - val_binary_accuracy: 1.0000
            10 - 186ms/step - loss: 0.3004 - binary_accuracy: 0.9296 - val_loss: 0.1373 - val_binary_accuracy: 1.0000
        """
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
        early_stopping = EarlyStopping(monitor='val_loss', patience=8, mode='min', restore_best_weights=True)
        res = self.model.fit(self.train_data, self.train_labels, epochs=epochs, shuffle=False,
                             batch_size=batch_size, validation_data=(self.val_data, self.val_labels),
                             callbacks=[reduce_lr, early_stopping],
                             verbose=0)

        self.predict_labels(self.val_data, self.val_labels)

    def predict_labels(self, data, labels):

        # Make predictions on test data
        predicted_labels = self.model.predict(data)

        cm = confusion_matrix(np.argmax(labels, axis=1), np.argmax(predicted_labels, axis=1))
        TP = cm[1][1]
        FP = cm[0][1]
        FN = cm[1][0]
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * (precision * recall) / (precision + recall)

        # print(f"F1 score: {f1} \nPrecision: {precision} \nRecall: {recall} \nConfusion Matrix :\n{cm}")

    @staticmethod
    def decoder(encoded_lists):
        reverse_mapping = {0: "A", 1: "G", 2: "C", 3: "T", 4: "0"}
        motifs = []

        for encoded_array in encoded_lists:
            decoded_sequence = ''.join(reverse_mapping[np.argmax(encoded_array)] for encoded_array in encoded_array)
            motifs.append(decoded_sequence)
        return motifs

    def report_summary(self):
        """ Prints a summary of the model"""
        self.model.summary()

