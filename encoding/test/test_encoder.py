import unittest
from os import path

from featureextraction.src.dataparsing import DataParser
from encoding.src.encoder import ModelFactory, ModelType, Autoencoder


class ModelFactoryTests(unittest.TestCase):
    def test_simple_ae(self):
        """
        Tests if autoencoder model is build with the correct size fo input, output and encoding
        """
        factory = ModelFactory(2048, 128, 2048)
        model = factory.simple_ae()

        self.assertEqual([(None, 2048)], model.layers[0].input_shape)
        self.assertEqual((None, 128), model.layers[4].output_shape)
        self.assertEqual((None, 2048), model.layers[-1].output_shape)

    def test_factory_call(self):
        """
        Tests if the correct type of AE is returned in call method
        """
        model = ModelFactory(2048, 128, 2048)(ModelType.SIMPLE_AE)

        self.assertEqual([(None, 2048)], model.get_layer('ae_input').input_shape)
        self.assertEqual((None, 128), model.get_layer('code').output_shape)
        self.assertEqual((None, 2048), model.get_layer('ae_output').output_shape)


class AutoencoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Defines testing data
        """
        cls.x_train = DataParser.get_features(path.join('mockfiles', 'birds_x_train.txt'))
        cls.y_train = DataParser.get_labels(path.join('mockfiles', 'birds_y_train.txt'))
        cls.x_test = DataParser.get_features(path.join('mockfiles', 'birds_x_test.txt'))
        cls.y_test = DataParser.get_labels(path.join('mockfiles', 'birds_y_test.txt'))

    def test_define_classifier(self):
        """
        Tests if best SVM parameters are defined
        """
        ae = Autoencoder(ModelType.SIMPLE_AE, self.x_train.shape[1], 128, self.x_train.shape[1])
        ae.define_classifier(self.x_train, self.y_train, nfolds=2)
        self.assertIsNotNone(ae.svm_best_parameters)
        self.assertTrue(ae.svm_best_parameters['kernel'] == 'linear')
        self.assertTrue(ae.svm_best_parameters['C'] in [0.5, 1, 5, 10])

    def test_run_model(self):
        """
        Tests if best AE parameters and training history are defined
        """
        ae = Autoencoder(ModelType.SIMPLE_AE, self.x_train.shape[1], 128, self.x_train.shape[1])
        ae.run_ae_model(self.x_train, self.y_train, self.x_test, self.y_test, 5, nfolds=2)

        self.assertIsNotNone(ae.history)
        self.assertTrue(ae.best_accuracy[0] > 0)
        self.assertTrue(ae.best_accuracy[1] > -1)
        self.assertIsNotNone(ae.best_model_weights)

        for i in range(5):
            self.assertTrue(0 <= ae.accuracies['train'][i] <= 1)
            self.assertTrue(0 <= ae.accuracies['test'][i] <= 1)
            self.assertTrue(0 <= ae.history.history['val_loss'][i] <= 1)
            self.assertTrue(0 <= ae.history.history['loss'][i] <= 1)
