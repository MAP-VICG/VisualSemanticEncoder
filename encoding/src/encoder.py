from enum import Enum
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import LambdaCallback

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score


class ModelType(Enum):
    """
    Enum for model type
    """
    SIMPLE_AE = "SIMPLE_AE"


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


class Autoencoder:
    def __init__(self, ae_type, input_length, encoding_length, output_length):
        """
        Initializes autoencoder attributes and defines the autoencoder model

        @param ae_type: type of autoencoder
        @param input_length: length of input for autoencoder
        @param encoding_length: length of autoencoder's code
        @param output_length: length of output or autoencoder
        """
        self.noise = 0
        self.history = None
        self.accuracies = None
        self.best_accuracy = (0, -1)
        self.best_model_weights = None
        self.svm_best_parameters = None
        self.confusion_matrices = None
        self.tuning_params = {'kernel': ['linear'], 'C': [0.5, 1, 5, 10]}
        self.autoencoder = ModelFactory(input_length, encoding_length, output_length)(ae_type)

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
                                 iid=False, scoring='recall_macro', n_jobs=njobs)

        svm_model.fit(x_train, y_train)
        self.svm_best_parameters = svm_model.best_params_

        return svm_model.best_estimator_

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
            svm_model.fit(encoder.predict(x_train), y_train)
            prediction = svm_model.predict(encoder.predict(x_test))
            self.accuracies[epoch] = balanced_accuracy_score(prediction, y_test)
            self.confusion_matrices[epoch] = confusion_matrix(prediction, y_test)

            if self.accuracies[epoch] > self.best_accuracy[0]:
                self.best_accuracy = (self.accuracies[epoch], epoch)
                self.best_model_weights = self.autoencoder.weights

        self.accuracies = [None] * nepochs
        self.confusion_matrices = [None] * nepochs

        svm_model = self.define_classifier(x_train, y_train, nfolds, njobs)
        encoder = Model(self.autoencoder.input, outputs=[self.autoencoder.get_layer('code').output])
        classification = LambdaCallback(on_epoch_end=svm_callback)

        self.history = self.autoencoder.fit(x_train + self.noise, x_train, epochs=nepochs, batch_size=512, shuffle=True,
                                            verbose=1, validation_split=0.2, callbacks=[classification])
