import numpy as np
from enum import Enum
import keras.backend as K
from keras.models import Model, Sequential
from keras.losses import mse
from keras.callbacks import LambdaCallback
from keras.layers import Input, Dense, Dropout, Concatenate, Lambda, Multiply, Add, Layer

from encoding.src.tools import zsl_eval


class ModelType(Enum):
    """
    Enum for model type
    """
    CONCAT_AE = "CONCAT_AE"
    SEMANTIC_AE = "SEMANTIC_AE"
    CONCAT_VAE = "CONCAT_VAE"
    SEMANTIC_VAE = "SEMANTIC_VAE"


class KLDivergenceLayer(Layer):

    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs


class Autoencoder:
    def __init__(self, vis_length, sem_length, code_length):
        """
        Initializes autoencoder attributes and defines the autoencoder model

        @param vis_length: length of visual input for autoencoder
        @param sem_length: length of semantic input for autoencoder
        @param code_length: length of autoencoder's code
        """
        self.model = None
        self.history = None
        self.vis_length = vis_length
        self.sem_length = sem_length
        self.code_length = code_length
        self.zsl_accuracies = None

    def define_model(self):
        """
        Builds autoencoder model

        @return: None
        """
        pass

    def run_ae_model(self, x_train_vis, x_train_sem, num_epochs):
        """
        Trains autoencoder

        @param x_train_vis: 2D numpy array with training visual data
        @param x_train_sem: 2D numpy array with training semantic data
        @param num_epochs: number of epochs
        @return: None
        """
        pass


class ConcatAutoencoder(Autoencoder):
    def define_model(self):
        """
        Builds an extended version of simple autoencoder model with 3 encoding layers and 3 decoding layers
        where all of them are dense and use relu activation function. The optimizer is defined to be adam and the
        loss to be mse.

        @return: None
        """
        input_vis = Input(shape=(self.vis_length,), name='ae_input_vis')
        input_sem = Input(shape=(self.sem_length,), name='ae_input_sem')
        concat_fts = Concatenate(name='concat')([input_vis, input_sem])

        encoded = Dense(1826, activation='relu', name='e_dense1')(concat_fts)
        encoded = Dense(932, activation='relu', name='e_dense2')(encoded)
        encoded = Dense(428, activation='relu', name='e_dense3')(encoded)

        encoded = Dropout(0.1)(encoded)
        code = Dense(self.code_length, activation='relu', name='code')(encoded)

        decoded = Dense(428, activation='relu', name='d_dense4')(code)
        decoded = Dense(932, activation='relu', name='d_dense5')(decoded)
        decoded = Dense(1826, activation='relu', name='d_dense6')(decoded)

        output_fts = Dense(self.vis_length + self.sem_length, activation='relu', name='ae_output')(decoded)

        self.model = Model(inputs=[input_vis, input_sem], outputs=output_fts)
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae', 'acc'])

    def run_model(self, vis_fts, sem_fts, num_epochs):
        """
        Trains autoencoder

        @param vis_fts: 2D numpy array with training visual data
        @param sem_fts: 2D numpy array with training semantic data
        @param num_epochs: number of epochs
        @return: None
        """
        self.history = self.model.fit([vis_fts, sem_fts], np.hstack((vis_fts, sem_fts)),
                                      epochs=num_epochs, batch_size=128, shuffle=True, verbose=1, validation_split=0.2)


class SemanticAutoencoder(Autoencoder):
    def define_model(self):
        """
        Builds an extended version of simple autoencoder model with 3 encoding layers and 3 decoding layers
        where all of them are dense and use relu activation function. The optimizer is defined to be adam and the
        loss to be mse.

        @return: None
        """
        input_vis = Input(shape=(self.vis_length,), name='ae_input_vis')
        input_sem = Input(shape=(self.sem_length,), name='ae_input_sem')

        encoded = Dense(1826, activation='relu', name='e_dense1')(input_vis)
        encoded = Dense(932, activation='relu', name='e_dense2')(encoded)
        encoded = Dense(428, activation='relu', name='e_dense3')(encoded)

        encoded = Dropout(0.1)(encoded)
        code = Dense(self.sem_length, activation='relu', name='code')(encoded)

        decoded = Dense(428, activation='relu', name='d_dense4')(code)
        decoded = Dense(932, activation='relu', name='d_dense5')(decoded)
        decoded = Dense(1826, activation='relu', name='d_dense6')(decoded)

        output_fts = Dense(self.vis_length, activation='relu', name='ae_output')(decoded)

        self.model = Model(inputs=[input_vis, input_sem], outputs=output_fts)
        self.model.add_loss(K.mean(K.square(input_sem - code)))
        self.model.add_loss(K.mean(K.square(input_vis - output_fts)))

        self.model.compile(optimizer='adam')

    def run_model(self, vis_fts, sem_fts, num_epochs, **kwargs):
        """
        Trains autoencoder

        @param vis_fts: 2D numpy array with training visual data
        @param sem_fts: 2D numpy array with training semantic data
        @param num_epochs: number of epochs
        @return: None
        """
        def zsl_callback(epoch, logs):
            """
            Runs ZSL and saves classification results
            @param epoch: default callback parameter. Epoch index.
            @param logs: default callback parameter. Loss result.
            """
            estimate = encoder.predict([kwargs['x_test_vis'], np.ones((kwargs['x_test_vis'].shape[0],
                                                                       kwargs['sem_length']))])

            zsl_accuracies[epoch], _ = zsl_eval(estimate, kwargs['labels_map'], 1, kwargs['testclasses_id'],
                                                kwargs['test_labels'])

        zsl_accuracies = [0] * num_epochs
        encoder = Model(self.model.input, outputs=[self.model.get_layer('code').output])
        classification = LambdaCallback(on_epoch_end=zsl_callback)

        self.history = self.model.fit([vis_fts, sem_fts], epochs=num_epochs, batch_size=128,
                                      shuffle=True, verbose=1, validation_split=0.2, callbacks=[classification])
        self.zsl_accuracies = zsl_accuracies


class ConcatVAE(Autoencoder):
    @staticmethod
    def sampling(args):
        """Reparameterization trick by sampling from an isotropic unit Gaussian.

        # Arguments
            args (tensor): mean and log of variance of Q(z|X)

        # Returns
            z (tensor): sampled latent vector
        """

        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def define_model(self):
        """
        Builds an extended version of simple autoencoder model with 3 encoding layers and 3 decoding layers
        where all of them are dense and use relu activation function. The optimizer is defined to be adam and the
        loss to be mse.

        @return: None
        """
        input_fts = Input(shape=(self.vis_length + self.sem_length,), name='ae_input')

        encoded = Dense(1826, activation='relu', name='e_dense1')(input_fts)
        encoded = Dense(932, activation='relu', name='e_dense2')(encoded)
        encoded = Dense(428, activation='relu', name='e_dense3')(encoded)

        encoded = Dropout(0.1)(encoded)
        z_mean = Dense(self.sem_length, name='z_mean')(encoded)
        z_log_var = Dense(self.sem_length, name='z_log_var')(encoded)

        z = Lambda(ConcatVAE.sampling, output_shape=(self.sem_length,), name='code')([z_mean, z_log_var])

        decoded = Dense(428, activation='relu', name='d_dense4')(z)
        decoded = Dense(932, activation='relu', name='d_dense5')(decoded)
        decoded = Dense(1826, activation='relu', name='d_dense6')(decoded)
        output_fts = Dense(self.vis_length + self.sem_length, activation='relu', name='ae_output')(decoded)

        vae = Model(input_fts, output_fts, name='vae_mlp')

        reconstruction_loss = mse(input_fts, output_fts)
        reconstruction_loss *= (self.vis_length + self.sem_length)
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        vae.compile(optimizer='adam')

        self.model = vae

    def run_model(self, vis_fts, sem_fts, num_epochs, **kwargs):
        """
        Trains autoencoder

        @param vis_fts: 2D numpy array with training visual data
        @param sem_fts: 2D numpy array with training semantic data
        @param num_epochs: number of epochs
        @return: None
        """
        def zsl_callback(epoch, logs):
            """
            Runs ZSL and saves classification results
            @param epoch: default callback parameter. Epoch index.
            @param logs: default callback parameter. Loss result.
            """
            estimate = encoder.predict(np.hstack((kwargs['x_test_vis'], kwargs['x_test_sem'])))

            zsl_accuracies[epoch], _ = zsl_eval(estimate, kwargs['labels_map'], 1, kwargs['testclasses_id'],
                                                kwargs['test_labels'])

        zsl_accuracies = [0] * num_epochs
        encoder = Model(self.model.input, outputs=[self.model.get_layer('code').output])
        classification = LambdaCallback(on_epoch_end=zsl_callback)

        self.history = self.model.fit(np.hstack((vis_fts, sem_fts)), epochs=num_epochs, batch_size=128,
                                      shuffle=True, verbose=1, validation_split=0.2, callbacks=[classification])
        self.zsl_accuracies = zsl_accuracies


class SemanticVAE(Autoencoder):
    @staticmethod
    def nll(y_true, y_pred):
        """ Negative log likelihood (Bernoulli). """

        # keras.losses.binary_crossentropy gives the mean
        # over the last axis. we require the sum
        return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)

    def define_model(self):
        """
        Builds an extended version of simple autoencoder model with 3 encoding layers and 3 decoding layers
        where all of them are dense and use relu activation function. The optimizer is defined to be adam and the
        loss to be mse.

        @return: None
        """
        epsilon_std = 1.0
        input_vis = Input(shape=(self.vis_length + self.sem_length,), name='vis_input')
        input_sem = Input(shape=(self.sem_length,), name='sem_input')

        encoded = Dense(1826, activation='relu', name='e_dense1')(input_vis)
        encoded = Dense(932, activation='relu', name='e_dense2')(encoded)
        encoded = Dense(428, activation='relu', name='e_dense3')(encoded)

        encoded = Dropout(0.1)(encoded)
        z_mu = Dense(self.sem_length)(encoded)
        z_log_var = Dense(self.sem_length)(encoded)

        z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
        z_sigma = Lambda(lambda t: K.exp(.5 * t))(z_log_var)

        eps = Input(tensor=K.random_normal(stddev=epsilon_std, shape=K.shape(input_sem)))
        # eps = Input(tensor=K.random_normal(stddev=epsilon_std, shape=(K.shape(input)[0], self.sem_length)))

        z_eps = Multiply()([z_sigma, eps])
        code = Add(name='code')([z_mu, z_eps])

        decoded = Dense(428, activation='relu', name='d_dense4')(code)
        decoded = Dense(932, activation='relu', name='d_dense5')(decoded)
        decoded = Dense(1826, activation='relu', name='d_dense6')(decoded)
        output_fts = Dense(self.vis_length + self.sem_length, activation='relu', name='ae_output')(decoded)

        self.model = Model(inputs=[input_vis, input_sem, eps], outputs=output_fts)

        self.model.add_loss(SemanticVAE.nll)
        self.model.add_loss(K.mean(K.square(input_sem - code)))
        self.model.add_loss(K.mean(K.square(input_vis - output_fts)))

        self.model.compile(optimizer='rmsprop')
        # self.model.compile(optimizer='rmsprop', loss=SemanticVAE.nll)

    def run_model(self, vis_fts, sem_fts, num_epochs, **kwargs):
        """
        Trains autoencoder

        @param vis_fts: 2D numpy array with training visual data
        @param sem_fts: 2D numpy array with training semantic data
        @param num_epochs: number of epochs
        @return: None
        """
        def zsl_callback(epoch, logs):
            """
            Runs ZSL and saves classification results
            @param epoch: default callback parameter. Epoch index.
            @param logs: default callback parameter. Loss result.
            """
            estimate = encoder.predict(np.hstack((kwargs['x_test_vis'], kwargs['x_test_sem'])))

            zsl_accuracies[epoch], _ = zsl_eval(estimate, kwargs['labels_map'], 1, kwargs['testclasses_id'],
                                                kwargs['test_labels'])

        zsl_accuracies = [0] * num_epochs
        encoder = Model(self.model.input, outputs=[self.model.get_layer('code').output])
        classification = LambdaCallback(on_epoch_end=zsl_callback)

        self.history = self.model.fit(np.hstack((vis_fts, sem_fts)), epochs=num_epochs, batch_size=128, shuffle=True,
                                      verbose=1, validation_split=0.2, callbacks=[classification])
        self.zsl_accuracies = zsl_accuracies


class ModelFactory:
    def __init__(self, vis_length, sem_length, code_length):
        """
        Initializes autoencoder attributes and defines the autoencoder model

        @param vis_length: length of visual input for autoencoder
        @param sem_length: length of semantic input for autoencoder
        @param code_length: length of autoencoder's code
        """
        self.vis_length = vis_length
        self.sem_length = sem_length
        self.code_length = code_length

    def __call__(self, ae_type):
        """
        Builds an autoencoder model based on the given type

        @param ae_type: autoencoder type
        @return: object with autoencoder model
        """
        if ae_type == ModelType.CONCAT_AE:
            model = ConcatAutoencoder(self.vis_length, self.sem_length, self.code_length)
            model.define_model()

            return model

        if ae_type == ModelType.SEMANTIC_AE:
            model = SemanticAutoencoder(self.vis_length, self.sem_length, self.code_length)
            model.define_model()

            return model

        if ae_type == ModelType.CONCAT_VAE:
            model = ConcatVAE(self.vis_length, self.sem_length, self.code_length)
            model.define_model()

            return model

        if ae_type == ModelType.SEMANTIC_VAE:
            model = SemanticVAE(self.vis_length, self.sem_length, self.code_length)
            model.define_model()

            return model
