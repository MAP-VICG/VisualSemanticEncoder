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
import numpy as np
from enum import Enum

from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.layers import Input, Dense, Dropout


class ModelType(Enum):
    """
    Enum for model type
    """
    SIMPLE_AE = "SIMPLE_AE"
    SIMPLE_VAE = "SIMPLE_VAE"


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

    def simple_ae(self):
        """
        Builds a simple autoencoder model with 3 encoding layers and 3 decoding layers where all of them are
        dense and use relu activation function. The optimizer is defined to be adam and the loss to be mse.

        @return: object with autoencoder model
        """
        input_fts = Input(shape=(self.input_length,), name='ae_input')

        encoded = Dense(self.input_length - round(0.3 * self.input_length), activation='relu', name='e_dense1')(input_fts)
        encoded = Dense(self.input_length - round(0.6 * self.input_length), activation='relu', name='e_dense2')(encoded)
        encoded = Dense(self.input_length - round(0.9 * self.input_length), activation='relu', name='e_dense3')(encoded)

        encoded = Dropout(0.2)(encoded)
        code = Dense(self.encoding_length, activation='relu', name='code')(encoded)

        decoded = Dense(self.input_length - round(0.9 * self.input_length), activation='relu', name='d_dense4')(code)
        decoded = Dense(self.input_length - round(0.6 * self.input_length), activation='relu', name='d_dense5')(decoded)
        decoded = Dense(self.input_length - round(0.3 * self.input_length), activation='relu', name='d_dense6')(decoded)
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
        pass


class Encoder:
    def __init__(self, input_length, encoding_length, output_length, ae_type, epochs):
        """
        Instantiates model object and initializes auxiliary variables

        :param input_length: length of AE input
        :param encoding_length: length of AE code
        :param output_length:  length of AE output
        :param ae_type: type of AE
        :param epochs: number of epochs to run training
        """
        self.epochs = epochs
        self.model = ModelFactory(input_length, encoding_length, output_length)(ae_type)
        self.encoder = Model(self.model.input, outputs=[self.model.get_layer('code').output])
        self.history = dict()

    def _fit(self, tr_vis_data, tr_sem_data):
        """
        Trains AE model for the specified number of epochs based on the input data. In each epoch,
        the classification accuracy is computed for the training set. Input and output of the model
        is the concatenation of the visual data with the semantic data.

        :param tr_vis_data: 2D numpy array with training visual data
        :param tr_sem_data: 2D numpy array with training semantic data
        :return: None
        """
        def ae_callback(epoch, logs):
            if logs['loss'] < self.history['best_loss'][0]:
                self.history['best_loss'] = (logs['loss'], epoch)
                self.history['best_weights'] = self.model.get_weights()

        self.history['best_weights'] = None
        self.history['best_loss'] = (float('inf'), 0)

        x_train = np.hstack((tr_vis_data, tr_sem_data))
        result = self.model.fit(x_train, x_train, epochs=self.epochs, batch_size=256, shuffle=True, verbose=1,
                                validation_split=0.2, callbacks=[LambdaCallback(on_epoch_end=ae_callback)])

        self.history['loss'] = result.history['loss']
        self.history['val_loss'] = result.history['val_loss']

    def estimate_semantic_data(self, tr_vis_data, te_vis_data, tr_sem_data, te_sem_data):
        """
        Estimates semantic data for the test set based on the best model computed in the AE training.

        :param tr_vis_data: training visual data
        :param te_vis_data: test visual data
        :param tr_sem_data: training semantic data
        :param te_sem_data: test semantic data
        :return: tuple with 2D numpy arrays with the computed semantic data for training and test sets
        """
        self._fit(tr_vis_data, tr_sem_data)
        self.model.set_weights(self.history['best_weights'])
        self.encoder = Model(self.model.input, outputs=[self.model.get_layer('code').output])

        return self.encoder.predict(np.hstack((tr_vis_data, tr_sem_data))), \
               self.encoder.predict(np.hstack((te_vis_data, te_sem_data)))
