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
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

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

        encoded = Dense(1826, activation='relu', name='e_dense1')(input_fts)
        encoded = Dense(932, activation='relu', name='e_dense2')(encoded)
        encoded = Dense(428, activation='relu', name='e_dense3')(encoded)

        encoded = Dropout(0.2)(encoded)
        code = Dense(self.encoding_length, activation='relu', name='code')(encoded)

        decoded = Dense(428, activation='relu', name='d_dense4')(code)
        decoded = Dense(932, activation='relu', name='d_dense5')(decoded)
        decoded = Dense(1826, activation='relu', name='d_dense6')(decoded)
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


class Autoencoder:
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
        self.history = {'loss': [], 'val_loss': [], 'acc': [], 'best_accuracy': [], 'best_model_weights': []}

    def _fit(self, tr_vis_data, tr_sem_data, tr_labels):
        """
        Trains AE model for the specified number of epochs based on the input data. In each epoch,
        the classification accuracy is computed for the training set. Input and output of the model
        is the concatenation of the visual data with the semantic data.

        :param tr_vis_data: 2D numpy array with training visual data
        :param tr_sem_data: 2D numpy array with training semantic data
        :param tr_labels: training labels
        :return: None
        """
        def svm_callback(epoch, logs):
            clf.fit(self.encoder.predict(x_train), tr_labels)
            prediction = clf.predict(self.encoder.predict(x_train))
            accuracies[epoch] = balanced_accuracy_score(prediction, tr_labels)

            if accuracies[epoch] > self.history['best_accuracy'][-1][0]:
                self.history['best_accuracy'][-1] = (accuracies[epoch], epoch)
                self.history['best_model_weights'][-1] = self.model.get_weights()

        self.history['best_accuracy'].append((0, 0))
        self.history['best_model_weights'].append(None)
        accuracies = np.zeros(self.epochs)

        x_train = np.hstack((tr_vis_data, tr_sem_data))
        clf = make_pipeline(StandardScaler(), SVC(gamma='auto', C=1.0, kernel='linear'))
        classification = LambdaCallback(on_epoch_end=svm_callback)

        result = self.model.fit(x_train, x_train, epochs=self.epochs, batch_size=256, shuffle=True, verbose=1,
                                validation_split=0.2, callbacks=[classification])

        self.history['loss'].append(result.history['loss'])
        self.history['val_loss'].append(result.history['val_loss'])
        self.history['acc'].append(accuracies)

    def estimate_semantic_data(self, tr_vis_data, tr_sem_data, te_vis_data, te_sem_data, tr_labels):
        """
        Estimates semantic data for the test set based on the best model computed in the AE training.

        :param tr_vis_data: training visual data
        :param tr_sem_data: training semantic data
        :param te_vis_data: test visual data
        :param te_sem_data: test semantic data
        :param tr_labels: training labels
        :return: 2D numpy array with the computed semantic data
        """
        self._fit(tr_vis_data, tr_sem_data, tr_labels)
        self.model.set_weights(self.history['best_model_weights'][-1])
        self.encoder = Model(self.model.input, outputs=[self.model.get_layer('code').output])

        return self.encoder.predict(np.hstack((te_vis_data, te_sem_data)))
