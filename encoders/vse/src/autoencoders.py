"""
Has methods to define, train and predict different types of autoencoders.

@author: Damares Resende
@contact: damaresresende@usp.br
@since: June 16, 2020

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""
import os
import json
import numpy as np

from tensorflow.keras import backend
from tensorflow.keras.losses import mse
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.layers import Input, Dense, Concatenate

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score


class StraightAutoencoder:
    def __init__(self, input_length, encoding_length, output_length):
        """
        Initializes main variables

        @param input_length: length of input for auto encoder
        @param encoding_length: length of auto encoder's code
        @param output_length: length of output or auto encoder
        """
        self.ae = None
        self.history = dict()
        self.best_weights = None
        self.input_length = input_length
        self.output_length = output_length
        self.encoding_length = encoding_length

    def define_ae(self):
        """
        Builds an autoencoder with 3 encoding layers and 3 decoding layers. Layers are dense and use ReLu activation
        function. The optimizer is defined to be Adam and the loss to be MSE. Inputs is the visual data and semantic
        data concatenated in one single 2D array, where rows are different examples and columns the attributes
        definition.

        @return: None
        """
        input_fts = Input(shape=(self.input_length,), name='ae_input')

        reduction_factor = (self.input_length - 2 * self.encoding_length)
        encoded = Dense(self.input_length - round(0.3 * reduction_factor), activation='relu', name='e_dense1')(input_fts)
        encoded = Dense(self.input_length - round(0.6 * reduction_factor), activation='relu', name='e_dense2')(encoded)
        encoded = Dense(self.input_length - round(0.9 * reduction_factor), activation='relu', name='e_dense3')(encoded)

        code = Dense(self.encoding_length, activation='relu', name='code')(encoded)

        decoded = Dense(self.input_length - round(0.9 * reduction_factor), activation='relu', name='d_dense4')(code)
        decoded = Dense(self.input_length - round(0.6 * reduction_factor), activation='relu', name='d_dense5')(decoded)
        decoded = Dense(self.input_length - round(0.3 * reduction_factor), activation='relu', name='d_dense6')(decoded)
        output_fts = Dense(self.output_length, activation='relu', name='ae_output')(decoded)

        self.ae = Model(inputs=input_fts, outputs=output_fts)
        self.ae.compile(optimizer='adam', loss='mse')

    def fit(self, tr_data, tr_labels, te_data, te_labels, epochs, results_path='.', save_weights=False):
        """
        Trains AE model for the specified number of epochs based on the input data. In each epoch,
        the classification accuracy is computed for the training and test sets. Input and output of the model
        is the concatenation of the visual data with the semantic data.

        :param tr_data: 2D numpy array with training visual and semantic data
        :param tr_labels: 1D numpy array with training labels
        :param te_data: 2D numpy array with test visual and semantic data
        :param te_labels: 1D numpy array with test labels
        :param epochs: number of epochs to train model
        :param results_path: string with path to save results
        :param save_weights: if True, saves training best weights
        :return: None
        """
        if results_path and not os.path.isdir(results_path):
            os.makedirs(results_path)

        def ae_callback(epoch, logs):
            if logs['loss'] < self.history['best_loss'][0]:
                self.best_weights = self.ae.get_weights()
                self.history['best_loss'] = (logs['loss'], epoch)

                if save_weights:
                    self.ae.save_weights(os.path.join(results_path, 'ae_model.h5'))

            encoder = Model(self.ae.input, outputs=[self.ae.get_layer('code').output])
            svm = make_pipeline(StandardScaler(), SVC(gamma='auto', C=1.0, kernel='linear'))

            svm.fit(encoder.predict(tr_data), tr_labels)
            self.history['svm_train'].append(balanced_accuracy_score(tr_labels, svm.predict(encoder.predict(tr_data))))
            self.history['svm_test'].append(balanced_accuracy_score(te_labels, svm.predict(encoder.predict(te_data))))

        if self.ae is None:
            self.define_ae()

        self.history['svm_train'] = []
        self.history['svm_test'] = []
        self.history['best_loss'] = (float('inf'), 0)

        result = self.ae.fit(tr_data, tr_data, epochs=epochs, batch_size=256, shuffle=True, verbose=1,
                             validation_data=(te_data, te_data), callbacks=[LambdaCallback(on_epoch_end=ae_callback)])

        self.history['loss'] = list(result.history['loss'])
        self.history['val_loss'] = list(result.history['val_loss'])

        if results_path and os.path.isdir(results_path):
            with open(os.path.join(results_path, 'ae_training_history.json'), 'w+') as f:
                json.dump(self.history, f, indent=4, sort_keys=True)

    def predict(self, tr_vis_data, tr_sem_data, te_vis_data, te_sem_data):
        """
        Projects training and test data in a new subspace using the best encoder built during training

        :param tr_vis_data: 2D numpy array with training visual data
        :param tr_sem_data: 2D numpy array with training semantic data
        :param te_vis_data: 2D numpy array with test visual data
        :param te_sem_data: 2D numpy array with test semantic data
        """
        if self.ae and self.best_weights:
            self.ae.set_weights(self.best_weights)
            encoder = Model(self.ae.input, outputs=[self.ae.get_layer('code').output])
            tr_est = encoder.predict(np.hstack((tr_vis_data, tr_sem_data)))
            te_est = encoder.predict(np.hstack((te_vis_data, te_sem_data)))

            return tr_est, te_est

        raise AttributeError('Cannot estimate values before training the model. Please run fit function first.')


class BalancedAutoencoder:
    def __init__(self, input_length, encoding_length, output_length):
        """
        Initializes main variables

        @param input_length: length of input for auto encoder
        @param encoding_length: length of auto encoder's code
        @param output_length: length of output or auto encoder
        """
        self.ae = None
        self.history = dict()
        self.best_weights = None
        self.input_length = input_length
        self.output_length = output_length
        self.encoding_length = encoding_length

    def define_ae(self):
        """
        Builds an autoencoder with 3 encoding layers and 3 decoding layers. Layers are dense and use ReLu activation
        function. The optimizer is defined to be Adam and loss as the MSE of visual features plus the MSE of semantic
        features. Inputs is the visual data and semantic data separately into two 2D array, where rows are different
        examples and columns the attributes definition.

        @return: None
        """
        input_vis = Input(shape=(self.input_length - self.encoding_length,), name='ae_input_vis')
        input_sem = Input(shape=(self.encoding_length,), name='ae_input_sem')
        concat = Concatenate(name='input_concat')([input_vis, input_sem])

        lambda_ = self.input_length / self.encoding_length
        reduction_factor = (self.input_length - 2 * self.encoding_length)
        encoded = Dense(self.input_length - round(0.3 * reduction_factor), activation='relu', name='e_dense1')(concat)
        encoded = Dense(self.input_length - round(0.6 * reduction_factor), activation='relu', name='e_dense2')(encoded)
        encoded = Dense(self.input_length - round(0.9 * reduction_factor), activation='relu', name='e_dense3')(encoded)

        code = Dense(self.encoding_length, activation='relu', name='code')(encoded)

        decoded = Dense(self.input_length - round(0.9 * reduction_factor), activation='relu', name='d_dense4')(code)
        decoded = Dense(self.input_length - round(0.6 * reduction_factor), activation='relu', name='d_dense5')(decoded)
        decoded = Dense(self.input_length - round(0.3 * reduction_factor), activation='relu', name='d_dense6')(decoded)

        output_sem = Dense(self.encoding_length, activation='relu', name='ae_output_sem')(decoded)
        output_vis = Dense(self.output_length - self.encoding_length, activation='relu', name='ae_output_vis')(decoded)

        self.ae = Model(inputs=[input_vis, input_sem], outputs=[output_sem, output_vis])
        loss = backend.mean(mse(output_vis, input_vis) + lambda_ * mse(output_sem, input_sem))

        self.ae.add_loss(loss)
        self.ae.compile(optimizer='adam')

    def fit(self, tr_data, tr_labels, te_data, te_labels, epochs, results_path='.', save_weights=False):
        """
        Trains AE model for the specified number of epochs based on the input data. In each epoch, the classification
        accuracy is computed for the training and test sets. Input of the model are the visual and semantic data in
        separate arrays. Output is the concatenation of the visual data with the semantic data.

        :param tr_data: 2D numpy array with training visual and semantic data
        :param tr_labels: 1D numpy array with training labels
        :param te_data: 2D numpy array with test visual and semantic data
        :param te_labels: 1D numpy array with test labels
        :param epochs: number of epochs to train model
        :param results_path: string with path to save results
        :param save_weights: if True, saves training best weights
        :return: None
        """
        if results_path and results_path != '.' and not os.path.isdir(results_path):
            os.makedirs(results_path)

        def ae_callback(epoch, logs):
            if logs['loss'] < self.history['best_loss'][0]:
                self.best_weights = self.ae.get_weights()
                self.history['best_loss'] = (logs['loss'], epoch)

                if save_weights:
                    self.ae.save_weights(os.path.join(results_path, 'ae_model.h5'))

            encoder = Model(self.ae.input, outputs=[self.ae.get_layer('code').output])
            svm = make_pipeline(StandardScaler(), SVC(gamma='auto', C=1.0, kernel='linear'))
            svm.fit(encoder.predict([tr_data[:, :limit], tr_data[:, limit:]]), tr_labels)

            prediction = svm.predict(encoder.predict([te_data[:, :limit], te_data[:, limit:]]))
            self.history['svm_train'].append(balanced_accuracy_score(te_labels, prediction))

            prediction = svm.predict(encoder.predict([te_data[:, :limit], te_data[:, limit:]]))
            self.history['svm_test'].append(balanced_accuracy_score(te_labels, prediction))

        if self.ae is None:
            self.define_ae()

        self.history['svm_train'] = []
        self.history['svm_test'] = []
        self.history['best_loss'] = (float('inf'), 0)

        limit = tr_data.shape[1] - self.encoding_length
        result = self.ae.fit([tr_data[:, :limit], tr_data[:, limit:]], [tr_data[:, :limit], tr_data[:, limit:]],
                             epochs=epochs, batch_size=256, shuffle=True,
                             verbose=1, callbacks=[LambdaCallback(on_epoch_end=ae_callback)],
                             validation_data=([te_data[:, :limit], te_data[:, limit:]], [te_data[:, :limit], te_data[:, limit:]]))

        self.history['loss'] = list(result.history['loss'])
        self.history['val_loss'] = list(result.history['val_loss'])

        if results_path and os.path.isdir(results_path):
            with open(os.path.join(results_path, 'ae_training_history.json'), 'w+') as f:
                json.dump(self.history, f, indent=4, sort_keys=True)

    def predict(self, tr_vis_data, tr_sem_data, te_vis_data, te_sem_data):
        """
        Projects training and test data in a new subspace using the best encoder built during training

        :param tr_vis_data: 2D numpy array with training visual data
        :param tr_sem_data: 2D numpy array with training semantic data
        :param te_vis_data: 2D numpy array with test visual data
        :param te_sem_data: 2D numpy array with test semantic data
        """
        if self.ae and self.best_weights:
            self.ae.set_weights(self.best_weights)
            encoder = Model(self.ae.input, outputs=[self.ae.get_layer('code').output])
            tr_est = encoder.predict([tr_vis_data, tr_sem_data])
            te_est = encoder.predict([te_vis_data, te_sem_data])

            return tr_est, te_est

        raise AttributeError('Cannot estimate values before training the model. Please run fit function first.')


class ZSLAutoencoder:
    def __init__(self, input_length, encoding_length, output_length):
        """
        Initializes main variables

        @param input_length: length of input for auto encoder
        @param encoding_length: length of auto encoder's code
        @param output_length: length of output or auto encoder
        """
        self.ae = None
        self.history = dict()
        self.best_weights = None
        self.input_length = input_length
        self.output_length = output_length
        self.encoding_length = encoding_length

    def define_ae(self):
        """
        Builds an autoencoder with 3 encoding layers and 3 decoding layers. Layers are dense and use ReLu activation
        function. The optimizer is defined to be Adamax and loss as the MSE of visual features plus the MSE of
        semantic features. Inputs is the visual data and semantic data separately into two 2D array, where rows are
        different examples and columns the attributes definition, however, in this case, the semantic data is not
        part of reconstruction, it defines the code, so the task changes from reconstruction the semantic and visual
        data to projecting the visual space to the semantic space.

        @return: None
        """
        input_vis = Input(shape=(self.input_length - self.encoding_length,), name='ae_input_vis')
        input_sem = Input(shape=(self.encoding_length,), name='ae_input_sem')

        lambda_ = self.input_length / self.encoding_length
        reduction_factor = (self.input_length - 2 * self.encoding_length)
        encoded = Dense(self.input_length - round(0.3 * reduction_factor), activation='relu', name='e_dense1')(input_vis)
        encoded = Dense(self.input_length - round(0.6 * reduction_factor), activation='relu', name='e_dense2')(encoded)
        encoded = Dense(self.input_length - round(0.9 * reduction_factor), activation='relu', name='e_dense3')(encoded)

        code = Dense(self.encoding_length, activation='relu', name='code')(encoded)

        decoded = Dense(self.input_length - round(0.9 * reduction_factor), activation='relu', name='d_dense4')(code)
        decoded = Dense(self.input_length - round(0.6 * reduction_factor), activation='relu', name='d_dense5')(decoded)
        decoded = Dense(self.input_length - round(0.3 * reduction_factor), activation='relu', name='d_dense6')(decoded)

        output_vis = Dense(self.output_length - self.encoding_length, activation='relu', name='ae_output_vis')(decoded)

        self.ae = Model(inputs=[input_vis, input_sem], outputs=output_vis)
        loss = 10000 * backend.mean(mse(input_vis, output_vis) + lambda_ * mse(input_sem, code))
        self.ae.add_loss(loss)

        self.ae.compile(optimizer='adamax')

    def fit(self, tr_vis_data, tr_sem_data, epochs, results_path='.', save_weights=False):
        """
        Trains AE model for the specified number of epochs based on the input data. Inputs of the model are the visual
        and semantic data in separate arrays, but only visual data is used for reconstruction. Encoder is set to define
        the semantic data from the visual data and decoder to define the visual data from the semantic data in the code.

        :param tr_vis_data: 2D numpy array with training visual data
        :param tr_sem_data: 12D numpy array with training semantic data
        :param epochs: number of epochs to train model
        :param results_path: string with path to save results
        :param save_weights: if True, saves training best weights
        :return: None
        """
        if results_path and results_path != '.' and not os.path.isdir(results_path):
            os.makedirs(results_path)

        def ae_callback(epoch, logs):
            if logs['loss'] < self.history['best_loss'][0]:
                self.best_weights = self.ae.get_weights()
                self.history['best_loss'] = (logs['loss'], epoch)

                if save_weights:
                    self.ae.save_weights(os.path.join(results_path, 'ae_model.h5'))

        if self.ae is None:
            self.define_ae()

        self.history['best_loss'] = (float('inf'), 0)

        result = self.ae.fit([tr_vis_data, tr_sem_data], tr_vis_data, epochs=epochs, batch_size=256,
                             shuffle=True, verbose=1, callbacks=[LambdaCallback(on_epoch_end=ae_callback)])

        self.history['loss'] = list(result.history['loss'])

        if results_path and os.path.isdir(results_path):
            with open(os.path.join(results_path, 'ae_training_history.json'), 'w+') as f:
                json.dump(self.history, f, indent=4, sort_keys=True)

    def predict(self, tr_vis_data, te_vis_data):
        """
        Projects training and test data in a new subspace using the best encoder built during training.
        This new subspace should was training to be the same as the semantic data

        :param tr_vis_data: 2D numpy array with training visual data
        :param te_vis_data: 2D numpy array with test visual data
        """
        if self.ae and self.best_weights:
            self.ae.set_weights(self.best_weights)
            encoder = Model(self.ae.input, outputs=[self.ae.get_layer('code').output])
            tr_est = encoder.predict([tr_vis_data, np.zeros((tr_vis_data.shape[0], self.encoding_length))])
            te_est = encoder.predict([te_vis_data, np.zeros((te_vis_data.shape[0], self.encoding_length))])

            return tr_est, te_est

        raise AttributeError('Cannot estimate values before training the model. Please run fit function first.')
