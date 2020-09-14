import os
import json
import numpy as np

from sklearn.svm import SVC
import tensorflow.keras.backend as K
from tensorflow.keras.losses import mse
from tensorflow.keras.models import Model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from tensorflow.keras.callbacks import LambdaCallback
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate


class SimpleAutoEncoder:
    def __init__(self, input_length, encoding_length, output_length):
        self.ae = None
        self.history = dict()
        self.best_weights = None
        self.input_length = input_length
        self.output_length = output_length
        self.encoding_length = encoding_length

    def define_ae(self):
        """
        Builds a simple auto encoder model with 1 encoding layer and 1 decoding layer. Layers are
        dense and use relu activation function. The optimizer is defined to be adam and the loss to be mse.

        @return: object with auto encoder model
        """
        input_fts = Input(shape=(self.input_length,), name='ae_input')

        reduction_factor = (self.input_length - 2 * self.encoding_length)
        encoded = Dense(self.input_length - round(0.3 * reduction_factor), activation='relu', name='e_dense1')(input_fts)
        encoded = Dense(self.input_length - round(0.6 * reduction_factor), activation='relu', name='e_dense2')(encoded)
        encoded = Dense(self.input_length - round(0.9 * reduction_factor), activation='relu', name='e_dense3')(encoded)

        code = Dense(self.encoding_length, activation='relu', name='code')(encoded)
        code = Dropout(0.1)(code)

        decoded = Dense(self.input_length - round(0.9 * reduction_factor), activation='relu', name='d_dense4')(code)
        decoded = Dense(self.input_length - round(0.6 * reduction_factor), activation='relu', name='d_dense5')(decoded)
        decoded = Dense(self.input_length - round(0.3 * reduction_factor), activation='relu', name='d_dense6')(decoded)
        output_fts = Dense(self.output_length, activation='relu', name='ae_output')(decoded)

        ae = Model(inputs=input_fts, outputs=output_fts)
        ae.compile(optimizer='adam', loss='mse')

        return ae

    def fit(self, tr_data, tr_labels, te_data, te_labels, epochs, results_path='.', save_weights=False):
        """
        Trains AE model for the specified number of epochs based on the input data. In each epoch,
        the classification accuracy is computed for the training set. Input and output of the model
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

            svm.fit(encoder.predict(x_train), y_train)
            self.history['svm_val'].append(balanced_accuracy_score(y_val, svm.predict(encoder.predict(x_val))))
            self.history['svm_test'].append(balanced_accuracy_score(te_labels, svm.predict(encoder.predict(te_data))))

        self.ae = self.define_ae()
        self.history['svm_val'] = []
        self.history['svm_test'] = []
        self.history['best_loss'] = (float('inf'), 0)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
        train_index, val_index = next(sss.split(tr_data, tr_labels))
        x_train, x_val = tr_data[train_index], tr_data[val_index]
        y_train, y_val = tr_labels[train_index], tr_labels[val_index]

        result = self.ae.fit(x_train, x_train, epochs=epochs, batch_size=256, shuffle=True, verbose=1,
                             validation_data=(x_val, x_val), callbacks=[LambdaCallback(on_epoch_end=ae_callback)])

        self.history['loss'] = list(result.history['loss'])
        self.history['val_loss'] = list(result.history['val_loss'])

        if results_path and os.path.isdir(results_path):
            with open(os.path.join(results_path, 'ae_training_history.json'), 'w+') as f:
                json.dump(self.history, f, indent=4, sort_keys=True)

    def predict(self, tr_vis_data, tr_sem_data, te_vis_data, te_sem_data):
        if self.ae and self.best_weights:
            self.ae.set_weights(self.best_weights)
            encoder = Model(self.ae.input, outputs=[self.ae.get_layer('code').output])
            tr_est = encoder.predict(np.hstack((tr_vis_data, tr_sem_data)))
            te_est = encoder.predict(np.hstack((te_vis_data, te_sem_data)))

            return tr_est, te_est

        raise AttributeError('Cannot estimate values before training the model. Please run fit function first.')


class ConcatAutoEncoder:
    def __init__(self, input_length, encoding_length, output_length):
        self.ae = None
        self.history = dict()
        self.best_weights = None
        self.input_length = input_length
        self.output_length = output_length
        self.encoding_length = encoding_length

    def define_ae(self):
        """
        Builds a simple auto encoder model with 3 encoding layers and 3 decoding layers. Layers are
        dense and use relu activation function. Visual and semantic data are provided separately
        and concatenated inside the model. The optimizer is defined to be adam and the loss to be
        a weighted mse.

        @return: object with auto encoder model
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
        code = Dropout(0.1)(code)

        decoded = Dense(self.input_length - round(0.9 * reduction_factor), activation='relu', name='d_dense4')(code)
        decoded = Dense(self.input_length - round(0.6 * reduction_factor), activation='relu', name='d_dense5')(decoded)
        decoded = Dense(self.input_length - round(0.3 * reduction_factor), activation='relu', name='d_dense6')(decoded)

        output_vis = Dense(self.output_length - self.encoding_length, activation='relu', name='ae_output_vis')(decoded)
        output_sem = Dense(self.encoding_length, activation='relu', name='ae_output_sem')(decoded)

        ae = Model(inputs=[input_vis, input_sem], outputs=[output_vis, output_sem])
        loss = K.mean(mse(output_vis, input_vis) + lambda_ * mse(output_sem, input_sem))

        ae.add_loss(loss)
        ae.compile(optimizer='adam')

        return ae

    def fit(self, tr_data, tr_labels, te_data, te_labels, epochs, results_path='.', save_weights=False):
        """
        Trains AE model for the specified number of epochs based on the input data. In each epoch,
        the classification accuracy is computed for the training set. Input and output of the model
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
            svm.fit(encoder.predict([x_train[:, :limit], x_train[:, limit:]]), y_train)

            prediction = svm.predict(encoder.predict([x_val[:, :limit], x_val[:, limit:]]))
            self.history['svm_val'].append(balanced_accuracy_score(y_val, prediction))

            prediction = svm.predict(encoder.predict([te_data[:, :limit], te_data[:, limit:]]))
            self.history['svm_test'].append(balanced_accuracy_score(te_labels, prediction))

        self.ae = self.define_ae()
        self.history['svm_val'] = []
        self.history['svm_test'] = []
        self.history['best_loss'] = (float('inf'), 0)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
        train_index, val_index = next(sss.split(tr_data, tr_labels))
        x_train, x_val = tr_data[train_index], tr_data[val_index]
        y_train, y_val = tr_labels[train_index], tr_labels[val_index]
        limit = x_train.shape[1] - self.encoding_length

        result = self.ae.fit([x_train[:, :limit], x_train[:, limit:]], [x_train[:, :limit], x_train[:, limit:]],
                             epochs=epochs, batch_size=256, shuffle=True,
                             verbose=1, callbacks=[LambdaCallback(on_epoch_end=ae_callback)],
                             validation_data=([x_val[:, :limit], x_val[:, limit:]], [x_val[:, :limit], x_val[:, limit:]]))

        self.history['loss'] = list(result.history['loss'])
        self.history['val_loss'] = list(result.history['val_loss'])

        if results_path and os.path.isdir(results_path):
            with open(os.path.join(results_path, 'ae_training_history.json'), 'w+') as f:
                json.dump(self.history, f, indent=4, sort_keys=True)

    def predict(self, tr_vis_data, tr_sem_data, te_vis_data, te_sem_data):
        if self.ae and self.best_weights:
            self.ae.set_weights(self.best_weights)
            encoder = Model(self.ae.input, outputs=[self.ae.get_layer('code').output])
            tr_est = encoder.predict([tr_vis_data, tr_sem_data])
            te_est = encoder.predict([te_vis_data, te_sem_data])

            return tr_est, te_est

        raise AttributeError('Cannot estimate values before training the model. Please run fit function first.')


class ZSLAutoEncoder:
    def __init__(self, input_length, encoding_length, output_length):
        self.ae = None
        self.history = dict()
        self.best_weights = None
        self.input_length = input_length
        self.output_length = output_length
        self.encoding_length = encoding_length

    def define_ae(self):
        input_vis = Input(shape=(self.input_length - self.encoding_length,), name='ae_input_vis')
        input_sem = Input(shape=(self.encoding_length,), name='ae_input_sem')

        lambda_ = self.input_length / self.encoding_length
        reduction_factor = (self.input_length - 2 * self.encoding_length)
        encoded = Dense(self.input_length - round(0.3 * reduction_factor), activation='relu', name='e_dense1')(input_vis)
        encoded = Dense(self.input_length - round(0.6 * reduction_factor), activation='relu', name='e_dense2')(encoded)
        encoded = Dense(self.input_length - round(0.9 * reduction_factor), activation='relu', name='e_dense3')(encoded)

        code = Dense(self.encoding_length, activation='relu', name='code')(encoded)
        dropout = Dropout(0.1)(code)

        decoded = Dense(self.input_length - round(0.9 * reduction_factor), activation='relu', name='d_dense4')(dropout)
        decoded = Dense(self.input_length - round(0.6 * reduction_factor), activation='relu', name='d_dense5')(decoded)
        decoded = Dense(self.input_length - round(0.3 * reduction_factor), activation='relu', name='d_dense6')(decoded)

        output_vis = Dense(self.output_length - self.encoding_length, activation='relu', name='ae_output_vis')(decoded)

        ae = Model(inputs=[input_vis, input_sem], outputs=output_vis)
        loss = 10000 * K.mean(mse(input_vis, output_vis) + lambda_ * mse(input_sem, code))
        ae.add_loss(loss)

        ae.compile(optimizer='adamax')

        return ae

    def fit(self, tr_vis_data, tr_sem_data, epochs, results_path='.', save_weights=False):
        if results_path and results_path != '.' and not os.path.isdir(results_path):
            os.makedirs(results_path)

        def ae_callback(epoch, logs):
            if logs['loss'] < self.history['best_loss'][0]:
                self.best_weights = self.ae.get_weights()
                self.history['best_loss'] = (logs['loss'], epoch)

                if save_weights:
                    self.ae.save_weights(os.path.join(results_path, 'ae_model.h5'))

        self.ae = self.define_ae()
        self.history['best_loss'] = (float('inf'), 0)

        result = self.ae.fit([tr_vis_data, tr_sem_data], tr_vis_data, epochs=epochs, batch_size=256,
                             shuffle=True, verbose=1, callbacks=[LambdaCallback(on_epoch_end=ae_callback)])

        self.history['loss'] = list(result.history['loss'])

        if results_path and os.path.isdir(results_path):
            with open(os.path.join(results_path, 'ae_training_history.json'), 'w+') as f:
                json.dump(self.history, f, indent=4, sort_keys=True)

    def predict(self, tr_vis_data, te_vis_data):
        if self.ae and self.best_weights:
            self.ae.set_weights(self.best_weights)
            encoder = Model(self.ae.input, outputs=[self.ae.get_layer('code').output])
            tr_est = encoder.predict([tr_vis_data, np.zeros((tr_vis_data.shape[0], self.encoding_length))])
            te_est = encoder.predict([te_vis_data, np.zeros((te_vis_data.shape[0], self.encoding_length))])

            return tr_est, te_est

        raise AttributeError('Cannot estimate values before training the model. Please run fit function first.')
