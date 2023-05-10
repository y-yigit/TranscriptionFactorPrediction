#!usr/bin/env python3

import tensorflow
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv1D, Masking, LSTM, Dense, MaxPooling1D, Flatten, Dropout, BatchNormalization, Embedding, Bidirectional, GlobalMaxPool1D, InputLayer, GaussianNoise
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2, l1
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

        if model_name == "model_sigme54_weights":
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
        test_set_fraction = 0.5  # 80% for training, 20% for testing
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
                           optimizer="adam",#tensorflow.keras.optimizers.Adam(learning_rate=0.001),
                           metrics=['binary_accuracy'])#, SensitivityAtSpecificity(0.9)])

    def train_model(self, epochs, batch_size):
        """ Train and test the model

        Results epochs:
            1 - 187ms/step - loss: 0.6527 - binary_accuracy: 0.9237 - val_loss: 0.1032 - val_binary_accuracy: 1.0000
            10 - 186ms/step - loss: 0.3004 - binary_accuracy: 0.9296 - val_loss: 0.1373 - val_binary_accuracy: 1.0000
        """
        res = self.model.fit(self.train_data, self.train_labels, epochs=epochs, shuffle=True,
                             batch_size=batch_size, validation_data=(self.val_data, self.val_labels), verbose=0)
        print(res)


    def predict_labels(self):

        # Make predictions on test data
        predicted_labels = self.model.predict(self.test_data)

        cm = confusion_matrix(np.argmax(self.test_labels, axis=1), np.argmax(predicted_labels, axis=1))
        TP = cm[1][1]
        FP = cm[0][1]
        FN = cm[1][0]
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * (precision * recall) / (precision + recall)

        print(f"F1 score: {f1} \nPrecision: {precision} \nRecall: {recall} \nConfusion Matrix :\n{cm}")

     #   # Map the predicted label index to the corresponding label name using your label mapping
      #  label_mapping = {0: "Non-motif", 1: "Motif"}
       # pred_labels = [label_mapping[idx] for idx in pred_labels_idx]

        # Combine the predicted labels with the input data points into a list of tuples
        #results = list(zip(self.test_data, pred_labels))

        # Print the predicted labels along with their corresponding input data
       # for data, label in results:
        #    pass#print(f"Data: {self.decoder(data)}, Label: {label}")

    @staticmethod
    def decoder(encoded_lists):
        reverse_mapping = {0: "A", 1: "G", 2: "C", 3: "T"}
        motif = ""
        for encoded_list in encoded_lists:
            for index, encoded_nucleotide in enumerate(encoded_list):
                if encoded_nucleotide == 1:
                    nucleotide = reverse_mapping[index]
                    motif += nucleotide
        return motif

    def report_summary(self):
        """ Prints a summary of the model"""
        self.model.summary()

