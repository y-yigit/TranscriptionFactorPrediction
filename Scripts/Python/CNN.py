#!usr/bin/env python3

import tensorflow
from tensorflow import keras
from tensorflow.keras import regularizers, optimizers, losses
from tensorflow.keras.callbacks import ModelCheckpoint, History, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Conv1D, Masking, LSTM, Dense, MaxPooling1D, Flatten, Dropout, BatchNormalization, Embedding, Bidirectional, GlobalMaxPool1D, InputLayer, GaussianNoise, AveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2, l1
import keras_tuner as kt
from keras_tuner import RandomSearch, HyperParameters
from sklearn.utils import class_weight

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
        
        if model_name == "cnn_lstm_yy":
            model.add(Masking(mask_value=0))
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
            # Extra layer
            # bin_accuracy 0.4855 >>  0.5842, values smaller than 16 lead to a higher accuracy, look up why
            # loss 2.2998 >>  1,1451
            model.add(Dense(16, activation='relu', name='dense'))
            model.add(Dense(2, activation='softmax', name='prediction'))
            self.model = model
            
    def lstm_builder(self, hp):

        hp_units_lstm = hp.Int('units_lstm', min_value=8, max_value=48, step=8)
        hp_units_conv1 = hp.Int('units_conv', min_value=8, max_value=40, step=8)
        hp_units_conv2 = hp.Int('units_conv', min_value=16, max_value=64, step=16)
        hp_units_dense = hp.Int('units_dense', min_value=8, max_value=48, step=8)

        hp_learning_rate = hp.Choice('learning_rate', values=[1e-4, 1e-5])
        hp_kernel_size1 = hp.Choice('kernel_size', values=[8, 10, 12, 14])
        hp_kernel_size2 = hp.Choice('kernel_size', values=[7, 9, 11, 13])
        hp_kernel_size3 = hp.Choice('kernel_size', values=[6, 8, 10, 12])
        hp_kernel_size4 = hp.Choice('kernel_size', values=[4, 6, 8, 10])
        hp_pool_type = hp.Choice('pool_type', values=['max', 'average'])

        hp_lstm_activation1 = hp.Choice('lstm_activation', values=['relu', "swish", "mish", 'tanh', 'sigmoid'])
        hp_lstm_activation2 = hp.Choice('lstm_activation', values=['relu', "swish", "mish", 'tanh', 'sigmoid'])
        hp_lstm_activation3 = hp.Choice('lstm_activation', values=['relu', "swish", "mish", 'tanh', 'sigmoid'])
        hp_lstm_activation4 = hp.Choice('lstm_activation', values=['relu', "swish", "mish", 'tanh', 'sigmoid'])
        hp_lstm_activation5 = hp.Choice('lstm_activation', values=['relu', "swish", "mish", 'tanh', 'sigmoid'])
        hp_dropout_rate1 = hp.Float('dropout_rate', min_value=0.3, max_value=0.7, step=0.1)
        hp_dropout_rate2 = hp.Float('dropout_rate', min_value=0.3, max_value=0.7, step=0.1)
        hp_dropout_rate3 = hp.Float('dropout_rate', min_value=0.3, max_value=0.7, step=0.1)

        # New
        hp_pool_size = hp.Choice('pool_size', values=[2, 4])
        hp_pool_padding = hp.Choice('pool_padding', values=['valid', 'same'])


        model = keras.Sequential()
        model.add(Masking(mask_value=0))
        model.add(Conv1D(filters=hp_units_conv1, kernel_size=hp_kernel_size1, activation=hp_lstm_activation1, input_shape=self.input_shape, name='conv1d_1'))
        model.add(Conv1D(filters=hp_units_conv1, kernel_size=hp_kernel_size2, activation=hp_lstm_activation2, input_shape=self.input_shape, name='conv1d_2'))
        model.add(BatchNormalization(name='batchnorm_1'))
        model.add(MaxPooling1D(pool_size=2, name='max_pooling'))

        model.add(Dropout(hp_dropout_rate1, name='dropout_1'))
        model.add(Conv1D(filters=hp_units_conv2, kernel_size=hp_kernel_size3, activation=hp_lstm_activation3, input_shape=self.input_shape, name='conv1d_3'))
        model.add(Conv1D(filters=hp_units_conv2, kernel_size=hp_kernel_size4, activation=hp_lstm_activation4, input_shape=self.input_shape, name='conv1d_4'))
        model.add(BatchNormalization(name='batchnorm_2'))
        model.add(Dropout(hp_dropout_rate2, name='dropout_2'))
        model.add(LSTM(units=hp_units_lstm, return_sequences=True, recurrent_dropout=hp_dropout_rate3, name='lstm'))
        model.add(Flatten(name='flatten'))
        model.add(Dense(hp_units_dense, activation=hp_lstm_activation5, name='dense'))
        model.add(Dense(2, activation='softmax', name='prediction'))

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      loss='binary_crossentropy',
                      metrics=['accuracy', keras.metrics.Precision()])
        return model


    def blstm_builder(self, hp):
        hp_units_lstm = hp.Int('units_lstm', min_value=8, max_value=32, step=8)
        hp_units_conv = hp.Int('units_conv', min_value=16, max_value=64, step=16)
        hp_units_dense = hp.Int('units_dense', min_value=8, max_value=32, step=8)

        hp_learning_rate = hp.Choice('learning_rate', values=[1e-4, 1e-5])
        hp_pool_size = hp.Choice('pool_size', values=[2, 4, 6])
        hp_kernel_size = hp.Choice('kernel_size', values=[4, 8, 12])
        hp_conv_layers = hp.Int('conv_layers', min_value=1, max_value=3)

        hp_lstm_activation = hp.Choice('lstm_activation', values=['relu', "swish", "mish", 'tanh', 'sigmoid'])
        hp_dropout_rate = hp.Float('dropout_rate', min_value=0.3, max_value=0.7, step=0.1)
        hp_pool_padding = hp.Choice('pool_padding', values=['valid', 'same'])

        model = keras.Sequential()
        model.add(Masking(mask_value=0))
        model.add(
            Bidirectional(LSTM(units=hp_units_lstm, return_sequences=True, name='lstm'),
                          input_shape=self.input_shape))
      #  for _ in range(hp_conv_layers):
        model.add(Conv1D(filters=hp_units_conv, kernel_size=hp_kernel_size, name='conv1d'))

        model.add(MaxPooling1D(pool_size=hp_pool_size, padding=hp_pool_padding, name='max_pooling'))
        model.add(Flatten(name='flatten'))
        model.add(Dropout(hp_dropout_rate))
        model.add(Dense(hp_units_dense, activation=hp_lstm_activation, name='dense'))
        model.add(Dense(2, activation='softmax', name='prediction'))

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      loss="binary_crossentropy",
                      metrics=['accuracy', keras.metrics.Precision()])
        return model

    def create_hyper_model(self):
        tuner = kt.Hyperband(self.lstm_builder,
                             objective='val_accuracy',
                             max_epochs=200,
                             factor=3,
                             directory='my_dir',
                             project_name='intro_to_kt')

        stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        tuner.search(self.train_data, self.train_labels, epochs=200, validation_data=(self.val_data, self.val_labels), callbacks=[stop_early])

        # Get the optimal hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=100)[0]

        # Build the model with the optimal hyperparameters and train it on the data for 100 epochs
        model = tuner.hypermodel.build(best_hps)
        history = model.fit(self.train_data, self.train_labels, shuffle=True, batch_size = 32, epochs=200, validation_data=(self.val_data, self.val_labels))

        val_acc_per_epoch = history.history['val_accuracy']
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1

        hypermodel = tuner.hypermodel.build(best_hps)
        #hypermodel.summary()
        # Retrain the model
        hypermodel.fit(self.train_data, self.train_labels, shuffle=True, batch_size = 128, epochs=best_epoch, validation_data=(self.val_data, self.val_labels))

        return hypermodel, history, best_epoch

    def set_input(self, train_data: object, validation_data, test_data, train_labels: object, validation_labels):
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
        #test_set_fraction = 0.8  # 80% for training, 20% for testing
        #train_size = int(len(train_data) * test_set_fraction)

        self.train_data = train_data
        self.train_labels = train_labels

        self.val_data = validation_data
        self.val_labels = validation_labels

        self.test_data = test_data
        #self.test_labels = test_labels

    def compile_model(self):
        """ Compile the model
        """
        self.model.compile(loss='binary_crossentropy',
                           optimizer=tensorflow.keras.optimizers.Adam(learning_rate=1e-05),
                           metrics=['binary_accuracy', keras.metrics.Precision()])#, SensitivityAtSpecificity(0.9)])

    def train_model(self, epochs, batch_size):
        """ Train and test the model

        Results epochs:
            1 - 187ms/step - loss: 0.6527 - binary_accuracy: 0.9237 - val_loss: 0.1032 - val_binary_accuracy: 1.0000
            10 - 186ms/step - loss: 0.3004 - binary_accuracy: 0.9296 - val_loss: 0.1373 - val_binary_accuracy: 1.0000
        """
        early_stopping = EarlyStopping(monitor='val_loss', patience=8, mode='min', restore_best_weights=True)
        res = self.model.fit(self.train_data, self.train_labels, epochs=epochs, shuffle=True,
                             batch_size=batch_size, validation_data=(self.val_data, self.val_labels),
                             callbacks=[early_stopping])

        return self.model.predict(self.test_data)

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

        #print(f"F1 score: {f1} \nPrecision: {precision} \nRecall: {recall} \nConfusion Matrix :\n{cm}")

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


