import numpy as np
from enum import Enum
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.callbacks import LambdaCallback

import random
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score


class ModelType(Enum):
    """
    Enum for model type
    """
    SIMPLE_AE = "SIMPLE_AE"
    EXTENDED_AE = "EXTENDED_AE"


class ModelFactory:
    def __init__(self, input_length, encoding_length, output_length):
        """
        Initializes main variables

        @param input_length: length of input for autoencoder
        @param encoding_length: length of autoencoder's code
        @param output_length: length of output or autoencoder
        """
        self.input_length = input_length
        self.output_length = output_length
        self.encoding_length = encoding_length

    def __call__(self, ae_type):
        """
        Builds an autoencoder model based on the given type

        @param ae_type: autoencoder type
        @return: object with autoencoder model
        """
        if ae_type == ModelType.SIMPLE_AE:
            return self.simple_ae()
        if ae_type == ModelType.EXTENDED_AE:
            return self.extended_ae()

    def simple_ae(self):
        """
        Builds a simple autoencoder model with 3 encoding layers and 3 decoding layers where all of them are
        dense and use relu activation function. The optimizer is defined to be adam and the loss to be mse.

        @return: object with autoencoder model
        """
        input_fts = Input(shape=(self.input_length,), name='ae_input')

        encoded = Dense(1426, activation='relu', name='e_dense1')(input_fts)
        encoded = Dense(732, activation='relu', name='e_dense2')(encoded)
        encoded = Dense(328, activation='relu', name='e_dense3')(encoded)

        code = Dense(self.encoding_length, activation='relu', name='code')(encoded)
        decoded = Dense(328, activation='relu', name='d_dense4')(code)

        decoded = Dense(732, activation='relu', name='d_dense5')(decoded)
        decoded = Dense(1426, activation='relu', name='d_dense6')(decoded)
        output_fts = Dense(self.output_length, activation='relu', name='ae_output')(decoded)

        ae = Model(inputs=input_fts, outputs=output_fts)
        ae.compile(optimizer='adam', loss='mse', metrics=['mae', 'acc'])

        return ae

    def extended_ae(self):
        """
        Builds an extended version of simple autoencoder model with 3 encoding layers and 3 decoding layers
        where all of them are dense and use relu activation function. The optimizer is defined to be adam and the
        loss to be mse.

        @return: object with autoencoder model
        """
        input_fts = Input(shape=(self.input_length,), name='ae_input')

        encoded = Dense(1826, activation='relu', name='e_dense1')(input_fts)
        encoded = Dense(932, activation='relu', name='e_dense2')(encoded)
        encoded = Dense(428, activation='relu', name='e_dense3')(encoded)

        encoded = Dropout(0.05)(encoded)
        code = Dense(self.encoding_length, activation='relu', name='code')(encoded)

        decoded = Dense(428, activation='relu', name='d_dense4')(code)
        decoded = Dense(932, activation='relu', name='d_dense5')(decoded)
        decoded = Dense(1826, activation='relu', name='d_dense6')(decoded)
        output_fts = Dense(self.output_length, activation='relu', name='ae_output')(decoded)

        ae = Model(inputs=input_fts, outputs=output_fts)
        ae.compile(optimizer='adam', loss='mse', metrics=['mae', 'acc'])

        return ae


class Autoencoder:
    def __init__(self, ae_type, input_length, encoding_length, output_length, baseline):
        """
        Initializes autoencoder attributes and defines the autoencoder model

        @param ae_type: type of autoencoder
        @param input_length: length of input for autoencoder
        @param encoding_length: length of autoencoder's code
        @param output_length: length of output or autoencoder
        """
        self.history = None
        self.accuracies = None
        self.svm_model = None
        self.ae_type = ae_type
        self.baseline = baseline
        self.best_accuracy = (0, -1)
        self.best_model_weights = None
        self.svm_best_parameters = None
        self.output_length = output_length
        self.tuning_params = {'kernel': ['linear'], 'C': [0.5, 1, 5, 10]}
        self.model = ModelFactory(input_length, encoding_length, output_length)(ae_type)

    def define_classifier(self, x_train, y_train, nfolds=5, njobs=None):
        """
        Runs grid search on the SVM to define its best parameter for the given training data. These parameters
        are going to define the SVM classifier used in autoencoder's training.

        @param x_train: 2D numpy array with training data
        @param y_train: 1D numpy array with labels
        @param nfolds: number of folds. Must be greater than 2 and smaller than the number of classes.
        @param njobs: number of jobs to run in parallel on Grid Search. If None, uses 1. If -1, uses the maximum number
            of threads available
        @return: object with Grid Search best model
        """
        svm_model = GridSearchCV(SVC(verbose=0, max_iter=1000, gamma='scale'), self.tuning_params, cv=nfolds,
                                 scoring='recall_macro', n_jobs=njobs)

        svm_model.fit(x_train, y_train)
        self.svm_best_parameters = svm_model.best_params_
        self.svm_model = svm_model.best_estimator_

    def define_best_models(self, x_train, y_train, weights_path):
        """
        Saves configuration of the best model after training.

        @param x_train: 2D numpy array with training set
        @param y_train: 1D numpy array with labels
        @param weights_path: path to save best model weights
        @return: None
        """
        self.model.set_weights(self.best_model_weights)
        encoder = Model(self.model.input, outputs=[self.model.get_layer('code').output])
        self.svm_model.fit(encoder.predict(x_train), y_train)
        self.model.save_weights(weights_path)

    @staticmethod
    def kill_semantic_attributes(data, rate):
        """
        Randomly sets to 0.1 a specific rate of the semantic attributes array

        @param data: 2D numpy array with data
        @param rate: float number from 0 to 1 specifying the rate of values to be set to 0.1
        @return: 2D numpy array with new data set
        """
        num_sem_attrs = abs(data.shape[1] - 2048)

        new_data = np.copy(data)
        for ex in range(new_data.shape[0]):
            mask = [False] * data.shape[1]
            for idx in random.sample(range(2048, data.shape[1]), round(num_sem_attrs * rate)):
                mask[idx] = True

            new_data[ex, mask] = new_data[ex, mask] * 0 + 0.1
        return new_data

    def run_ae_model(self, x_train, y_train, x_test, y_test, nepochs, nfolds=5, njobs=None):
        """
        Trains autoencoder and defines its best weights.

        @param x_train: 2D numpy array with training data
        @param y_train: 1D numpy array with labels
        @param x_test: 2D numpy array with test data
        @param y_test: 1D numpy array with labels
        @param nepochs: number of epochs
        @param nfolds: number of folds to be used in the SVM
        @param njobs: number of threads to be used in the SVM
        @return: None
        """
        def svm_callback(epoch, logs):
            """
            Runs SVM and saves prediction results

            @param epoch: default callback parameter. Epoch index.
            @param logs: default callback parameter. Loss result.
            """
            self.svm_model.fit(encoder.predict(x_train), y_train)
            prediction = self.svm_model.predict(encoder.predict(x_train))
            self.accuracies['train'][epoch] = balanced_accuracy_score(prediction, y_train)

            prediction = self.svm_model.predict(encoder.predict(x_test))
            self.accuracies['vs100 test'][epoch] = balanced_accuracy_score(prediction, y_test)

            x_test_tmp = np.copy(x_test)
            x_test_tmp[:, 2048:] = x_test_tmp[:, 2048:] * 0 + 0.1
            prediction = self.svm_model.predict(encoder.predict(x_test_tmp))
            self.accuracies['vis test'][epoch] = balanced_accuracy_score(prediction, y_test)

            x_test_tmp = np.copy(x_test)
            x_test_tmp[:, :2048] = x_test_tmp[:, :2048] * 0 + 0.1
            prediction = self.svm_model.predict(encoder.predict(x_test_tmp))
            self.accuracies['sem test'][epoch] = balanced_accuracy_score(prediction, y_test)

            x_test_tmp = Autoencoder.kill_semantic_attributes(x_test, 0.5)
            prediction = self.svm_model.predict(encoder.predict(x_test_tmp))
            self.accuracies['vs50 test'][epoch] = balanced_accuracy_score(prediction, y_test)

            if self.accuracies['vs100 test'][epoch] > self.best_accuracy[0]:
                self.best_accuracy = (self.accuracies['vs100 test'][epoch], epoch)
                self.best_model_weights = self.model.get_weights()

        self.accuracies = {'train': [None] * nepochs, 'vs100 test': [None] * nepochs, 'vis test': [None] * nepochs,
                           'sem test': [None] * nepochs, 'vs50 test': [None] * nepochs}

        self.define_classifier(x_train, y_train, nfolds, njobs)
        encoder = Model(self.model.input, outputs=[self.model.get_layer('code').output])
        classification = LambdaCallback(on_epoch_end=svm_callback)

        self.history = self.model.fit(x_train, x_train, epochs=nepochs, batch_size=128, shuffle=True, verbose=1,
                                      validation_split=0.2, callbacks=[classification])
