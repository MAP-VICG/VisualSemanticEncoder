import unittest
import numpy as np
from os import path

from featureextraction.src.dataparsing import DataIO
from encoding.src.encoder import ModelFactory, ModelType, Autoencoder


class ModelFactoryTests(unittest.TestCase):
    def test_simple_ae(self):
        """
        Tests if autoencoder model is build with the correct size fo input, output and encoding
        """
        factory = ModelFactory(2048, 128, 2048)
        model = factory.simple_ae()

        self.assertEqual((None, 2048), model.layers[0].input_shape)
        self.assertEqual((None, 128), model.layers[4].output_shape)
        self.assertEqual((None, 2048), model.layers[-1].output_shape)

    def test_factory_call(self):
        """
        Tests if the correct type of AE is returned in call method
        """
        model = ModelFactory(2048, 128, 2048)(ModelType.SIMPLE_AE)

        self.assertEqual((None, 2048), model.get_layer('ae_input').input_shape)
        self.assertEqual((None, 128), model.get_layer('code').output_shape)
        self.assertEqual((None, 2048), model.get_layer('ae_output').output_shape)


class AutoencoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Defines testing data
        """
        cls.x_train = DataIO.get_features(path.join('mockfiles', 'CUB200_x_train.txt'))
        cls.y_train = DataIO.get_labels(path.join('mockfiles', 'CUB200_y_train.txt'))
        cls.x_test = DataIO.get_features(path.join('mockfiles', 'CUB200_x_test.txt'))
        cls.y_test = DataIO.get_labels(path.join('mockfiles', 'CUB200_y_test.txt'))

    def test_define_classifier(self):
        """
        Tests if best SVM parameters are defined
        """
        ae = Autoencoder(ModelType.SIMPLE_AE, self.x_train.shape[1], 128, self.x_train.shape[1], {'vis', 0})
        ae.define_classifier(self.x_train, self.y_train, nfolds=2)
        self.assertIsNotNone(ae.svm_best_parameters)
        self.assertTrue(ae.svm_best_parameters['kernel'] == 'linear')
        self.assertTrue(ae.svm_best_parameters['C'] in [0.5, 1, 5, 10])

    def test_run_model(self):
        """
        Tests if best AE parameters and training history are defined
        """
        ae = Autoencoder(ModelType.SIMPLE_AE, self.x_train.shape[1], 128, self.x_train.shape[1], {'vis', 0})
        ae.run_ae_model(self.x_train, self.y_train, self.x_test, self.y_test, 5, nfolds=2)

        self.assertIsNotNone(ae.history)
        self.assertTrue(ae.best_accuracy[0] > 0)
        self.assertTrue(ae.best_accuracy[1] > -1)
        self.assertIsNotNone(ae.best_model_weights)

        for i in range(5):
            self.assertTrue(0 <= ae.accuracies['train'][i] <= 1)
            self.assertTrue(0 <= ae.accuracies['vis test'][i] <= 1)
            self.assertTrue(0 <= ae.accuracies['sem test'][i] <= 1)
            self.assertTrue(0 <= ae.accuracies['vs50 test'][i] <= 1)
            self.assertTrue(0 <= ae.accuracies['vs100 test'][i] <= 1)
            self.assertTrue(0 <= ae.history.history['val_loss'][i] <= 100)
            self.assertTrue(0 <= ae.history.history['loss'][i] <= 100)

    def test_kill_semantic_attributes(self):
        """
        Tests if semantic array is correctly destroyed
        """
        count = np.zeros(self.x_test.shape[0])
        for i, example in enumerate(self.x_test):
            for value in example:
                if value == 0.1:
                    count[i] += 1

        new_data = Autoencoder.kill_semantic_attributes(self.x_test, 0.5)

        new_data_count = np.zeros(self.x_test.shape[0])
        for i, example in enumerate(new_data):
            for value in example:
                if value == 0.1:
                    new_data_count[i] += 1

        x_test_count = np.zeros(self.x_test.shape[0])
        for i, example in enumerate(self.x_test):
            for value in example:
                if value == 0.1:
                    x_test_count[i] += 1

        self.assertTrue((x_test_count == count).all())
        self.assertTrue((new_data_count == count + 156).all())
