import unittest
from os import path

from featureextraction.src.dataparsing import DataIO
from encoding.src.encoder import ModelFactory, ModelType


class ModelFactoryTests(unittest.TestCase):
    def test_factory_call(self):
        """
        Tests if the correct type of AE is returned in call method
        """
        ae = ModelFactory(2048, 85, 128)(ModelType.CONCAT_AE)

        self.assertEqual((None, 2048), ae.model.get_layer('ae_input_vis').input_shape)
        self.assertEqual((None, 85), ae.model.get_layer('ae_input_sem').input_shape)
        self.assertEqual((None, 128), ae.model.get_layer('code').output_shape)
        self.assertEqual((None, 2048 + 85), ae.model.get_layer('ae_output').output_shape)


class AutoencoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Defines testing data
        """
        cls.x_train = DataIO.get_features(path.join('mockfiles', 'CUB200_x_train.txt'))

    def test_run_model(self):
        """
        Tests if best AE parameters and training history are defined
        """
        ae = ModelFactory(2048, 312, 128)(ModelType.CONCAT_AE)
        ae.run_model(vis_fts=self.x_train[:, :2048], sem_fts=self.x_train[:, 2048:], num_epochs=5)

        self.assertIsNotNone(ae.history)

        for i in range(5):
            self.assertTrue(0 <= ae.history.history['val_loss'][i] <= 100)
            self.assertTrue(0 <= ae.history.history['loss'][i] <= 100)
