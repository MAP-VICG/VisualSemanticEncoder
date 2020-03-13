from enum import Enum
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda, Layer, Multiply, Add
from tensorflow.keras.callbacks import LambdaCallback

from sklearn.svm import SVC
from keras import backend as K
from keras.losses import mse
from keras.metrics import accuracy
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score


class ModelType(Enum):
    """
    Enum for model type
    """
    SIMPLE_AE = "SIMPLE_AE"
    EXTENDED_AE = "EXTENDED_AE"
    SIMPLE_VAE = "SIMPLE_VAE"
    EXTENDED_VAE = "EXTENDED_VAE"


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
        if ae_type == ModelType.EXTENDED_VAE:
            return self.extended_vae()

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

        autoencoder = Model(inputs=input_fts, outputs=output_fts)
        autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae', 'acc'])

        return autoencoder

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

        autoencoder = Model(inputs=input_fts, outputs=output_fts)
        autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae', 'acc'])

        return autoencoder

    @staticmethod
    def _sampling(args):
        """
        Reparameterization trick by sampling from an isotropic unit Gaussian.

        @param args: tensor with mean and log of variance of Q(z|X)
        @return tensor with sampled latent vector
        """
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    @staticmethod
    def nll(y_true, y_pred):
        """ Negative log likelihood (Bernoulli). """

        # keras.losses.binary_crossentropy gives the mean
        # over the last axis. we require the sum
        return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)

    def extended_vae(self):
        epsilon_std = 1.0
        input_fts = Input(shape=(self.input_length,), name='ae_input')

        encoded = Dense(1826, activation='relu', name='e_dense1')(input_fts)
        encoded = Dense(932, activation='relu', name='e_dense2')(encoded)
        encoded = Dense(428, activation='relu', name='e_dense3')(encoded)

        encoded = Dropout(0.05)(encoded)
        z_mu = Dense(self.encoding_length)(encoded)
        z_log_var = Dense(self.encoding_length)(encoded)

        z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
        z_sigma = Lambda(lambda t: K.exp(.5 * t))(z_log_var)

        eps = Input(tensor=K.random_normal(stddev=epsilon_std,
                                           shape=(K.shape(input_fts)[0], self.encoding_length)))

        z_eps = Multiply()([z_sigma, eps])
        code = Add(name='code')([z_mu, z_eps])

        decoded = Dense(428, activation='relu', name='d_dense4')(code)
        decoded = Dense(932, activation='relu', name='d_dense5')(decoded)
        decoded = Dense(1826, activation='relu', name='d_dense6')(decoded)
        output_fts = Dense(self.output_length, activation='relu', name='ae_output')(decoded)

        vae = Model(inputs=[input_fts, eps], outputs=output_fts)
        vae.compile(optimizer='rmsprop', loss=ModelFactory.nll)

        return vae


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
        self.autoencoder = ModelFactory(2048, encoding_length, output_length)(ae_type)

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
        self.autoencoder.set_weights(self.best_model_weights)
        encoder = Model(self.autoencoder.input, outputs=[self.autoencoder.get_layer('code').output])
        self.svm_model.fit(encoder.predict(x_train[:, :2048]), y_train)
        self.autoencoder.save_weights(weights_path)

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
            self.svm_model.fit(encoder.predict(x_train[:, :2048]), y_train)
            prediction = self.svm_model.predict(encoder.predict(x_train[:, :2048]))
            self.accuracies['train'][epoch] = balanced_accuracy_score(prediction, y_train)

            prediction = self.svm_model.predict(encoder.predict(x_test[:, :2048]))
            self.accuracies['test'][epoch] = balanced_accuracy_score(prediction, y_test)

            if self.accuracies['test'][epoch] > self.best_accuracy[0]:
                self.best_accuracy = (self.accuracies['test'][epoch], epoch)
                self.best_model_weights = self.autoencoder.get_weights()

        self.accuracies = {'train': [None] * nepochs, 'test': [None] * nepochs}

        self.define_classifier(x_train, y_train, nfolds, njobs)
        encoder = Model(self.autoencoder.input, outputs=[self.autoencoder.get_layer('code').output])
        classification = LambdaCallback(on_epoch_end=svm_callback)

        # if self.ae_type in (ModelType.EXTENDED_VAE, ModelType.SIMPLE_VAE):
        #     self.history = self.autoencoder.fit(x_train, epochs=nepochs, batch_size=256, shuffle=True, verbose=1,
        #                                         validation_split=0.2, callbacks=[classification])
        # else:
        self.history = self.autoencoder.fit(x_train[:, :2048], x_train[:, :self.output_length], epochs=nepochs,
                                            batch_size=256, shuffle=True, verbose=1, validation_split=0.2,
                                            callbacks=[classification])
