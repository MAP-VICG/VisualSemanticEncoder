"""
Builds autoencoder that concatenates visual and semantic attributes by reducing the input
array dimensionality and creating a new feature space with the merged data.

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
from enum import Enum

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.layers import Input, Dense, Dropout, Layer, Concatenate

import tensorflow.keras.backend as K
from tensorflow.keras.losses import mse


class ModelType(Enum):
    """
    Enum for model type
    """
    SIMPLE_AE = "SIMPLE_AE"
    SIMPLE_VAE = "SIMPLE_VAE"
    CONCAT_AE = "CONCAT_AE"


class Sampling(Layer):
    """
    Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.
    """
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    def __init__(self, encoder, decoder, input_length, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.input_length = input_length

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *= self.input_length
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }


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
        if ae_type == ModelType.SIMPLE_VAE:
            return self.simple_vae()
        if ae_type == ModelType.CONCAT_AE:
            return self.concat_ae()

    def concat_ae(self):
        """
        Builds a simple autoencoder model with 3 encoding layers and 3 decoding layers where all of them are
        dense and use relu activation function. The optimizer is defined to be adam and the loss to be mse.

        @return: object with autoencoder model
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

        autoencoder = Model(inputs=[input_vis, input_sem], outputs=[output_vis, output_sem])
        loss = K.mean(mse(output_vis, input_vis) + lambda_ * mse(output_sem, input_sem))
        autoencoder.add_loss(loss)

        autoencoder.compile(optimizer='adam')

        return autoencoder

    def simple_ae(self):
        """
        Builds a simple autoencoder model with 3 encoding layers and 3 decoding layers where all of them are
        dense and use relu activation function. The optimizer is defined to be adam and the loss to be mse.

        @return: object with autoencoder model
        """
        input_fts = Input(shape=(self.input_length,), name='ae_input')

        reduction_factor = (self.input_length - 2 * self.encoding_length)
        encoded = Dense(self.input_length - round(0.4 * reduction_factor), activation='relu', name='e_dense1')(input_fts)
        encoded = Dense(self.input_length - round(0.8 * reduction_factor), activation='relu', name='e_dense2')(encoded)
        # encoded = Dense(self.input_length - round(0.9 * reduction_factor), activation='relu', name='e_dense3')(encoded)

        code = Dense(self.encoding_length, activation='relu', name='code')(encoded)
        code = Dropout(0.1)(code)

        # decoded = Dense(self.input_length - round(0.9 * reduction_factor), activation='relu', name='d_dense4')(code)
        decoded = Dense(self.input_length - round(0.8 * reduction_factor), activation='relu', name='d_dense5')(code)
        decoded = Dense(self.input_length - round(0.4 * reduction_factor), activation='relu', name='d_dense6')(decoded)
        output_fts = Dense(self.output_length, activation='relu', name='ae_output')(decoded)

        autoencoder = Model(inputs=input_fts, outputs=output_fts)
        autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae', 'acc'])

        return autoencoder

    def simple_vae(self):
        """
        Builds a simple variational autoencoder model with 3 encoding layers and 3 decoding layers, where all of them
        are dense and use relu activation function. The optimizer is defined to be adam and the loss to be mse.

        @return: object with VAE model
        """
        encoder_inputs = keras.Input(shape=(self.input_length,), name='ae_input')
        reduction_factor = (self.input_length - 2 * self.encoding_length)

        x = Dense(self.input_length - round(0.3 * reduction_factor), activation='relu')(encoder_inputs)
        x = Dense(self.input_length - round(0.6 * reduction_factor), activation='relu')(x)
        x = Dense(self.input_length - round(0.9 * reduction_factor), activation='relu')(x)

        z_mean = Dense(self.encoding_length, name="z_mean")(x)
        z_log_var = Dense(self.encoding_length, name="z_log_var")(x)
        z = Sampling(name="code")([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

        latent_inputs = keras.Input(shape=(self.encoding_length,))
        x = Dense(self.input_length - round(0.9 * reduction_factor), activation='relu')(latent_inputs)
        x = Dense(self.input_length - round(0.6 * reduction_factor), activation='relu')(x)
        x = Dense(self.input_length - round(0.3 * reduction_factor), activation='relu')(x)
        decoder_outputs = Dense(self.output_length, activation='relu', name='ae_output')(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

        vae = VAE(encoder, decoder, self.input_length)
        vae.compile(optimizer=keras.optimizers.Adam())
        return vae


class Encoder:
    def __init__(self, input_length, encoding_length, output_length, ae_type, epochs, results_path='.'):
        """
        Instantiates model object and initializes auxiliary variables

        :param input_length: length of AE input
        :param encoding_length: length of AE code
        :param output_length:  length of AE output
        :param ae_type: type of AE
        :param epochs: number of epochs to run training
        :param results_path: string with path to save results to
        """
        self.epochs = epochs
        self.model = ModelFactory(input_length, encoding_length, output_length)(ae_type)

        self.encoder = None
        self.ae_type = ae_type
        self.history = dict()
        self.best_weights = None
        self.results_path = results_path

    def _fit(self, tr_vis_data, tr_sem_data, save_results=False):
        """
        Trains AE model for the specified number of epochs based on the input data. In each epoch,
        the classification accuracy is computed for the training set. Input and output of the model
        is the concatenation of the visual data with the semantic data.

        :param tr_vis_data: 2D numpy array with training visual data
        :param tr_sem_data: 2D numpy array with training semantic data
        :param save_results: if True, saves training results
        :return: None
        """
        if save_results and self.results_path != '.' and not os.path.isdir(self.results_path):
            os.mkdir(self.results_path)

        def ae_callback(epoch, logs):
            if logs['loss'] < self.history['best_loss'][0]:
                self.best_weights = self.model.get_weights()
                self.history['best_loss'] = (logs['loss'], epoch)

                if save_results:
                    idx = str(epoch) if epoch > 9 else '0' + str(epoch)
                    self.model.save_weights(os.path.join(self.results_path, 'ae_model_%s.h5' % idx))

        self.best_weights = None
        self.history['best_loss'] = (float('inf'), 0)

        x_train = np.hstack((tr_vis_data, tr_sem_data))
        if self.ae_type == ModelType.SIMPLE_AE:
            result = self.model.fit(x_train, x_train, epochs=self.epochs, batch_size=256, shuffle=True, verbose=1,
                                    validation_split=0.2, callbacks=[LambdaCallback(on_epoch_end=ae_callback)])
            self.history['loss'] = list(result.history['loss'])
            self.history['val_loss'] = list(result.history['val_loss'])
        else:
            result = self.model.fit(x_train, x_train, epochs=self.epochs, batch_size=256, shuffle=True, verbose=1,
                                    callbacks=[LambdaCallback(on_epoch_end=ae_callback)])
            self.history['loss'] = list(result.history['loss'])
            self.history['kl_loss'] = list(result.history['kl_loss'])
            self.history['reconstruction_loss'] = list(result.history['reconstruction_loss'])

        if save_results:
            with open(os.path.join(self.results_path, 'ae_training_history.json'), 'w+') as f:
                json.dump(self.history, f, indent=4, sort_keys=True)

    def _fit_concat(self, tr_vis_data, tr_sem_data, save_results=False):
        """
        Trains AE model for the specified number of epochs based on the input data. In each epoch,
        the classification accuracy is computed for the training set. Input and output of the model
        is the concatenation of the visual data with the semantic data.

        :param tr_vis_data: 2D numpy array with training visual data
        :param tr_sem_data: 2D numpy array with training semantic data
        :param save_results: if True, saves training results
        :return: None
        """
        if save_results and self.results_path != '.' and not os.path.isdir(self.results_path):
            os.mkdir(self.results_path)

        def ae_callback(epoch, logs):
            if logs['loss'] < self.history['best_loss'][0]:
                self.best_weights = self.model.get_weights()
                self.history['best_loss'] = (logs['loss'], epoch)

                if save_results:
                    idx = str(epoch) if epoch > 9 else '0' + str(epoch)
                    self.model.save_weights(os.path.join(self.results_path, 'ae_model_%s.h5' % idx))

        self.best_weights = None
        self.history['best_loss'] = (float('inf'), 0)

        result = self.model.fit([tr_vis_data, tr_sem_data], [tr_vis_data, tr_sem_data], epochs=self.epochs,
                                batch_size=256, shuffle=True, verbose=1, validation_split=0.2,
                                callbacks=[LambdaCallback(on_epoch_end=ae_callback)])
        self.history['loss'] = list(result.history['loss'])
        self.history['val_loss'] = list(result.history['val_loss'])

        if save_results:
            with open(os.path.join(self.results_path, 'ae_training_history.json'), 'w+') as f:
                json.dump(self.history, f, indent=4, sort_keys=True)

    def estimate_semantic_data(self, tr_vis_data, te_vis_data, tr_sem_data, te_sem_data, save_results=False):
        """
        Estimates semantic data for the test set based on the best model computed in the AE training.

        :param tr_vis_data: training visual data
        :param te_vis_data: test visual data
        :param tr_sem_data: training semantic data
        :param te_sem_data: test semantic data
        :param save_results: if True, saves training results
        :return: tuple with 2D numpy arrays with the computed semantic data for training and test sets
        """
        if self.ae_type == ModelType.SIMPLE_AE:
            self._fit(tr_vis_data, tr_sem_data, save_results)
            self.model.set_weights(self.best_weights)

            self.encoder = Model(self.model.input, outputs=[self.model.get_layer('code').output])
            return self.encoder.predict(np.hstack((tr_vis_data, tr_sem_data))), \
                   self.encoder.predict(np.hstack((te_vis_data, te_sem_data)))

        elif self.ae_type == ModelType.SIMPLE_VAE:
            self._fit(tr_vis_data, tr_sem_data, save_results)
            self.model.set_weights(self.best_weights)

            self.encoder = self.model.encoder
            return self.encoder.predict(np.hstack((tr_vis_data, tr_sem_data)))[2], \
                   self.encoder.predict(np.hstack((te_vis_data, te_sem_data)))[2]

        else:
            self._fit_concat(tr_vis_data, tr_sem_data, save_results)
            self.model.set_weights(self.best_weights)

            self.encoder = Model(self.model.input, outputs=[self.model.get_layer('code').output])
            return self.encoder.predict([tr_vis_data, tr_sem_data]), \
                   self.encoder.predict([te_vis_data, te_sem_data])
