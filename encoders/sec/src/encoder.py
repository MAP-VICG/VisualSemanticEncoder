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
from enum import Enum

from encoders.sec.src.autoencoders import Simple3Layers, Concat3Layers


class ModelType(Enum):
    """
    Enum for model type
    """
    SIMPLE_AE = "SIMPLE_AE"
    CONCAT_AE = "CONCAT_AE"


class ModelFactory:
    def __init__(self, input_length, encoding_length, output_length):
        """
        Initializes main variables

        @param input_length: length of input for auto encoder
        @param encoding_length: length of auto encoder's code
        @param output_length: length of output or auto encoder
        """
        self.input_length = input_length
        self.output_length = output_length
        self.encoding_length = encoding_length

    def __call__(self, ae_type):
        """
        Builds an auto encoder model based on the given type

        @param ae_type: auto encoder type
        @return: object with auto encoder model
        """
        if ae_type == ModelType.SIMPLE_AE:
            return Simple3Layers(self.input_length, self.encoding_length, self.output_length)
        if ae_type == ModelType.CONCAT_AE:
            return Concat3Layers(self.input_length, self.encoding_length, self.output_length)


class Encoder:
    def __init__(self, input_length, encoding_length, output_length, ae_type, epochs, results_path='.'):
        """
        Instantiates model object and initializes auxiliary variables

        :param input_length: length of AE input
        :param encoding_length: length of AE code
        :param output_length:  length of AE output
        :param ae_type: type of auto encoder
        :param epochs: number of epochs to run training
        :param results_path: string with path to save results to
        """
        self.encoder = None
        self.epochs = epochs
        self.ae_type = ae_type
        self.results_path = results_path
        self.input_length = input_length
        self.output_length = output_length
        self.encoding_length = encoding_length

    def estimate_semantic_data(self, tr_vis_data, te_vis_data, tr_sem_data, te_sem_data, save_weights=False):
        """
        Estimates semantic data for the test set based on the best model computed in the AE training.

        :param tr_vis_data: training visual data
        :param te_vis_data: test visual data
        :param tr_sem_data: training semantic data
        :param te_sem_data: test semantic data
        :param save_weights: if True, saves training weights
        :return: tuple with 2D numpy arrays with the computed semantic data for training and test sets
        """
        model = ModelFactory(self.input_length, self.encoding_length, self.output_length)(self.ae_type)
        model.fit(tr_vis_data, tr_sem_data, self.epochs, self.results_path, save_weights)
        tr_est, te_est = model.predict(tr_vis_data, tr_sem_data, te_vis_data, te_sem_data)

        return tr_est, te_est
