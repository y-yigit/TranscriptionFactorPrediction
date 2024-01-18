#!usr/bin/env python3

"""
CCN.py

Module for CNN classification

__author__ = "Yaprak Yigit"
__version__ = "0.1"

.. note:: Use numpy 1.21 for compatibility with tensorflow on peregrine
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow
from sklearn.metrics import confusion_matrix

from tensorflow.keras import callbacks, layers, models, regularizers
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import Sequential, load_model
import keras_tuner as kt

# Callbacks
from tensorflow.keras.callbacks import ModelCheckpoint, History, ReduceLROnPlateau, EarlyStopping

# Layers
from tensorflow.keras.layers import (
    Conv1D, Masking, SimpleRNN, LSTM, Dense, MaxPooling1D, Flatten, Dropout, BatchNormalization, RandomZoom, Embedding,
    Bidirectional, GlobalMaxPool1D, Discretization, RandomCrop, InputLayer, Normalization, GaussianNoise,
    RandomTranslation, AveragePooling1D, GlobalAveragePooling1D, UpSampling1D, DepthwiseConv1D, LocallyConnected1D
)

# Regularizers
from tensorflow.keras.regularizers import l2, l1

# Define a learning rate scheduler function
def lr_scheduler(epoch, lr):
    """ Decaying learning rate
    After 10 epochs, decrease the learning rate every epoch

    :param epoch: Current epoch
    :param lr: Current learning rate
    """
    if epoch < 10:
        return lr
    else:
        return lr * 0.9

def plot_history(history, metric, test_specie):
    """ Plots a variable recorded in the model history

    :param history: History object
    :param metric: The metric to plot
    """
    # Plot training & validation loss values
    plt.plot(history.history[metric])
    plt.plot(history.history[f'val_{metric}'])
    plt.title(f'Loss vs Epoch for {test_specie.lower().replace("_", " ")}')
    plt.xlabel('Epoch')
    plt.ylabel("Loss")
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig(f'Output/Images/{metric}_{test_specie.lower()}_plot.png')

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

        :param input_shape: Tuple, representing the input shape of the data
        :param model_name: Name of the selected CNN model
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
            model.add(Dense(2, activation='softmax'))
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

        elif model_name == 'model_lstm_bi2':
            model.add(Masking(mask_value=0))
            model.add(Conv1D(filters=8, kernel_size=12, strides=1, activation='relu', input_shape=input_shape,
                             name='conv1d_1'))
            model.add(Conv1D(filters=8, kernel_size=9, strides=1, activation='relu', input_shape=input_shape,
                             name='conv1d_2'))
            model.add(BatchNormalization(name='batchnorm_1'))
            model.add(MaxPooling1D(pool_size=2, name='max_pooling'))
            model.add(Dropout(0.3, name='dropout_1'))
            model.add(Conv1D(filters=16, kernel_size=6, strides=1, activation='relu', input_shape=input_shape,
                             name='conv1d_3'))
            model.add(Conv1D(filters=16, kernel_size=4, strides=1, activation='relu', input_shape=input_shape,
                             name='conv1d_4'))
            model.add(BatchNormalization(name='batchnorm_2'))
            model.add(Dropout(0.3, name='dropout_2'))
            # model.add(MaxPooling1D(pool_size=2, name='max_pooling_2'))
            model.add(Bidirectional(LSTM(units=16, return_sequences=True, name='lstm'), input_shape=input_shape))
            model.add(Flatten(name='flatten'))
            model.add(Dense(16, activation='relu', name='dense'))
            model.add(Dense(2, activation='softmax', name='prediction'))
            self.model = model


        elif model_name == 'model_lstm_bi8':
            model.add(Masking(mask_value=0))
            model.add(Conv1D(filters=16, kernel_size=12, strides=1, activation='relu', input_shape=input_shape,
                             name='conv1d_1'))
            model.add(BatchNormalization(name='batchnorm_1'))
            model.add(Dropout(0.3, name='dropout_1'))
            model.add(Conv1D(filters=32, kernel_size=4, strides=1, activation='relu', input_shape=input_shape,
                             name='conv1d_4'))
            model.add(BatchNormalization(name='batchnorm_2'))
            model.add(Dropout(0.3, name='dropout_2'))
            model.add(Bidirectional(LSTM(units=16, return_sequences=True, name='lstm'), input_shape=input_shape))
            model.add(Flatten(name='flatten'))
            model.add(Dense(16, activation='relu', name='dense'))
            model.add(Dense(2, activation='softmax', name='prediction'))
            self.model = model


        elif model_name == "T":

            model.add(Masking(mask_value=0))
            model.add(Bidirectional(LSTM(units=32, return_sequences=True, activation='tanh',
                                         kernel_regularizer=l2(0.001), recurrent_activation='sigmoid', name='lstm')))
            model.add(Dropout(0.1))
            model.add(Conv1D(filters=16, kernel_size=12, name='conv1d', kernel_regularizer=l2(0.001)))
            model.add(MaxPooling1D(pool_size=4, name='max_pooling_2'))
            model.add(Flatten(name='flatten'))
            model.add(Dropout(0.5))
            model.add(Dense(16, activation='sigmoid', name='dense_3', kernel_regularizer=l2(0.001)))
            model.add(Dense(2, activation='softmax', name='prediction'))
            self.model = model

        elif model_name == 'model_lstm_bi6':
            model.add(Masking(mask_value=0))
            model.add(Conv1D(filters=16, kernel_size=12, strides=1, activation='relu', input_shape=input_shape,
                             name='conv1d_1'))
            model.add(BatchNormalization(name='batchnorm_1'))
            model.add(Dropout(0.3, name='dropout_1'))
            model.add(Bidirectional(LSTM(units=16, return_sequences=True, name='lstm'), input_shape=input_shape))
            model.add(Conv1D(filters=32, kernel_size=4, strides=1, activation='relu', input_shape=input_shape,
                             name='conv1d_4'))
            # model.add(BatchNormalization(name='batchnorm_2'))
            # model.add(MaxPooling1D(pool_size=2, name='max_pooling'))
            model.add(Dropout(0.3, name='dropout_2'))
            model.add(Bidirectional(LSTM(units=16, return_sequences=True, name='lstm'), input_shape=input_shape))
            model.add(Flatten(name='flatten'))
            model.add(Dense(16, activation='relu', name='dense'))
            model.add(Dense(2, activation='softmax', name='prediction'))
            self.model = model

        elif model_name == "base":
            model.add(Masking(mask_value=0))
            model.add(Conv1D(filters=16, kernel_size=12, name='conv1d', input_shape=input_shape))
            model.add(BatchNormalization())
            model.add(MaxPooling1D(pool_size=2, name='max_pooling'))
            model.add(Dropout(0.1))
            model.add(
                Bidirectional(LSTM(32,return_sequences=True, kernel_regularizer=l2(0.01)), input_shape=input_shape))

            model.add(Flatten(name='flatten'))
            model.add(Dropout(0.5))
            model.add(Dense(16, activation='relu', name='dense12'))
            model.add(Dense(2, activation='sigmoid'))
            self.model = model

        elif model_name == 'model_lstm_cnn_global':
            model.add(Masking(mask_value=0))
            model.add(LSTM(units=16, return_sequences=True, input_shape=input_shape, name='lstm'))
            model.add(Conv1D(filters=32, kernel_size=8, name='conv1d'))
            model.add(GlobalMaxPool1D(name='global_max_pooling'))
            model.add(Dense(16, activation='relu', name='dense'))
            model.add(Dense(2, activation='softmax', name='prediction'))
            self.model = model

        elif model_name == "cnn_new":
            model.add(Masking(mask_value=0))
            model.add(Conv1D(filters=16, kernel_size=12, name='conv1d', kernel_regularizer=l2(0.001), input_shape=input_shape))
            model.add(Conv1D(filters=16, kernel_size=9, name='conv12d', kernel_regularizer=l2(0.001), input_shape=input_shape))
            model.add(BatchNormalization())
            model.add(MaxPooling1D(pool_size=2, name='max_pooling'))
            model.add(Dropout(0.3, name='dropout_2'))
            model.add(Bidirectional(LSTM(16, return_sequences=True), input_shape=input_shape))
            model.add(Flatten(name='flatten'))
            model.add(Dense(16, activation='relu', name='dense12'))
            model.add(Dense(2, activation='sigmoid'))
            self.model = model

        elif model_name == "TBiNet":
            model.add(Masking(mask_value=0))
            # 2nd Layer: 1D Convolution Layer
            model.add(Conv1D(filters=16, kernel_size=8, activation='relu', padding='same'))
            model.add(Dropout(0.3))
            model.add(AveragePooling1D(pool_size=2, strides=2))
            # 5th Layer: Bidirectional LSTM Layer
            model.add(Bidirectional(LSTM(32, return_sequences=True)))
            model.add(GlobalAveragePooling1D())
            model.add(Dropout(0.2))
            model.add(Dense(16, activation='relu'))
            model.add(Dropout(0.15))
            model.add(Dense(16, activation='relu'))
            model.add(Dropout(0.05))
            model.add(Dense(2, activation='sigmoid'))
            self.model = model

    def blstm_builder(self, hp):
        """ Build a hypermodel with keras_tuner

        Redundant function because this does greatly improve the performance of the motif classifier
        """
        hp_units_lstm2 = hp.Int('units_lstm', min_value=16, max_value=128, step=16)
        hp_units_conv1 = hp.Int('units_conv', min_value=16, max_value=128, step=16)
        hp_units_conv2 = hp.Int('units_conv', min_value=16, max_value=128, step=16)
        hp_units_conv3 = hp.Int('units_conv', min_value=16, max_value=128, step=16)
        hp_units_conv4 = hp.Int('units_conv', min_value=16, max_value=128, step=16)
        hp_units_dense = hp.Int('units_dense', min_value=8, max_value=48, step=8)

        hp_learning_rate = hp.Choice('learning_rate', values=[1e-4, 1e-5])
        hp_kernel_size1 = hp.Choice('kernel_size', values=[4, 6, 8, 10, 12, 14])
        hp_kernel_size2 = hp.Choice('kernel_size', values=[7, 9, 11, 13])
        hp_kernel_size3 = hp.Choice('kernel_size', values=[4, 6, 8, 10, 12, 14])
        hp_kernel_size4 = hp.Choice('kernel_size', values=[4, 6, 8, 10, 12, 14])

        hp_lstm_activation1 = hp.Choice('lstm_activation', values=['relu', "swish", "mish", "sigmoid"])
        hp_lstm_activation2 = hp.Choice('lstm_activation', values=['relu', "swish", "mish", "sigmoid"])
        hp_lstm_activation3 = hp.Choice('lstm_activation', values=['relu', "swish", "mish", "sigmoid"])
        hp_lstm_activation4 = hp.Choice('lstm_activation', values=['relu', "swish", "mish", "sigmoid"])
        hp_lstm_activation5 = hp.Choice('lstm_activation', values=['relu', "swish", "mish", "sigmoid"])
        hp_lstm_activation6 = hp.Choice('lstm_activation', values=['relu', "swish", "mish", "sigmoid"])
        hp_dropout_rate1 = hp.Float('dropout_rate', min_value=0.0, max_value=0.7, step=0.1)
        hp_dropout_rate2 = hp.Float('dropout_rate', min_value=0.0, max_value=0.7, step=0.1)
        hp_dropout_rate3 = hp.Float('dropout_rate', min_value=0.0, max_value=0.7, step=0.1)
        hp_dropout_rate6 = hp.Float('dropout_rate', min_value=0.0, max_value=0.7, step=0.1)

        model = keras.Sequential()
        model.add(Masking(mask_value=0))

        model.add(Conv1D(filters=hp_units_conv1, kernel_size=hp_kernel_size1, activation=hp_lstm_activation1,
                         input_shape=self.input_shape, name=f'conv1d_1'))
        model.add(Conv1D(filters=hp_units_conv2, kernel_size=hp_kernel_size2, activation=hp_lstm_activation2,
                         input_shape=self.input_shape, name='conv1d_2'))
        model.add(BatchNormalization(name='batchnorm_1'))
        model.add(MaxPooling1D(pool_size=2, name='max_pooling'))

        model.add(Dropout(hp_dropout_rate1, name='dropout_1'))
        model.add(Bidirectional(
            LSTM(units=hp_units_lstm2, return_sequences=True, recurrent_dropout=hp_dropout_rate6, name='lstm'),
            input_shape=self.input_shape))
        model.add(Conv1D(filters=hp_units_conv3, kernel_size=hp_kernel_size3, activation=hp_lstm_activation3,
                         input_shape=self.input_shape, name='conv1d_3'))
        model.add(Conv1D(filters=hp_units_conv4, kernel_size=hp_kernel_size4, activation=hp_lstm_activation4,
                         input_shape=self.input_shape, name='conv1d_4'))
        model.add(BatchNormalization(name='batchnorm_2'))
        model.add(Dropout(hp_dropout_rate2, name='dropout_2'))
        model.add(Flatten(name='flatten'))
        model.add(Dense(hp_units_dense, activation=hp_lstm_activation5, name='dense'))
        model.add(Dense(2, activation=hp_lstm_activation6, name='prediction'))

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      loss='binary_crossentropy',
                      metrics=['f1_score', 'precision'])#'accuracy', keras.metrics.Precision()])
        return model

    def create_hyper_model(self):
        """ Call the hypermodel and train it"""
        tuner = kt.Hyperband(self.blstm_builder2,
                             objective='val_accuracy',
                             max_epochs=200,
                             factor=3,
                             directory='my_dir',
                             project_name='intro_to_kt')

        stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        tuner.search(self.train_data, self.train_labels, epochs=200, validation_data=(self.val_data, self.val_labels),
                     callbacks=[stop_early])

        # Get the optimal hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=100)[0]

        # Build the model with the optimal hyperparameters and train it on the data for 100 epochs
        model = tuner.hypermodel.build(best_hps)
        history = model.fit(self.train_data, self.train_labels, shuffle=True, batch_size=16, verbose=0, epochs=200,
                            validation_data=(self.val_data, self.val_labels))

        val_acc_per_epoch = history.history['val_accuracy']
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1

        hypermodel = tuner.hypermodel.build(best_hps)
        # hypermodel.summary()
        # Retrain the model
        hypermodel.fit(self.train_data, self.train_labels, shuffle=True, batch_size=16, verbose=0, epochs=best_epoch,
                       validation_data=(self.val_data, self.val_labels))

        return hypermodel, history, best_epoch

    def set_input(self, train_data: object, validation_data, test_data, train_labels: object, validation_labels):
        """ Sets the input data and labels
        """
        self.train_data = train_data
        self.train_labels = train_labels

        self.val_data = validation_data
        self.val_labels = validation_labels

        self.test_data = test_data

    def run_model(self, test_specie):
        """ Compile the model and return predictions
        """
        checkpoint_path = f'checkpoint.h5'
        if os.path.exists(checkpoint_path):
            self.model = load_model(checkpoint_path)
        else:
            self.model.compile(
                loss='binary_crossentropy',
                optimizer=tensorflow.keras.optimizers.Adam(learning_rate=1e-04),
                metrics=[tensorflow.keras.metrics.Precision(), tensorflow.keras.metrics.Recall()]
            )

        lr_scheduler_callback = LearningRateScheduler(lr_scheduler)
        model_checkpoint = ModelCheckpoint(checkpoint_path)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, start_from_epoch=5)

        history = self.model.fit(
            self.train_data, self.train_labels, epochs=50, shuffle=True,
            batch_size=16, validation_data=(self.val_data, self.val_labels),
            callbacks=[early_stopping, lr_scheduler_callback]#, model_checkpoint]
        )

        plot_history(history, "loss", test_specie)
        self.model.save(f"Output/Models/30-39_{test_specie}.keras")
        return self.model.predict(self.test_data)

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
