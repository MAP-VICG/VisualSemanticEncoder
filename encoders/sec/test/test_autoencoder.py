"""
Tests for module autoencoders

@author: Damares Resende
@contact: damaresresende@usp.br
@since: June 16, 2020

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""
import os
import unittest
import numpy as np
from scipy.io import loadmat

from encoders.sec.src.autoencoders import SimpleAutoEncoder, ConcatAutoEncoder, ZSLAutoEncoder


class AutoencodersTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Builds and train AE model
        """
        data = loadmat('../../../../Datasets/SAE/cub_demo_data.mat')
        labels_dict = {data['te_cl_id'][i][0]: attributes for i, attributes in enumerate(data['S_te_pro'])}
        s_te = np.array([labels_dict[label[0]] for label in data['test_labels_cub']])

        cls.code_length = data['S_tr'].shape[1]
        cls.test_labels = data['test_labels_cub']
        cls.train_labels = data['train_labels_cub']
        cls.test_data = np.hstack((data['X_te'], s_te))
        cls.train_data = np.hstack((data['X_tr'], data['S_tr']))

    def test_simple_ae(self):
        """
        Tests if simple autoencoder model is build with the correct size for input, output and encoding
        """
        model = SimpleAutoEncoder(2048 + 85, 85, 2048 + 85)
        model.define_ae()

        self.assertEqual([None, 2048 + 85], model.ae.input.shape.as_list())
        self.assertEqual([None, 2048 + 85], model.ae.output.shape.as_list())

        self.assertEqual([(None, 2048 + 85)], model.ae.get_layer('ae_input').input_shape)
        self.assertEqual((None, 85), model.ae.get_layer('code').output_shape)
        self.assertEqual((None, 2048 + 85), model.ae.get_layer('ae_output').output_shape)

    def test_concat_ae(self):
        """
        Tests if concat autoencoder model is build with the correct size for input, output and encoding
        """
        model = ConcatAutoEncoder(2048 + 85, 85, 2048 + 85)
        model.define_ae()

        self.assertEqual([None, 2048], model.ae.input[0].shape.as_list())
        self.assertEqual([None, 85], model.ae.input[1].shape.as_list())
        self.assertEqual([None, 85], model.ae.output[0].shape.as_list())
        self.assertEqual([None, 2048], model.ae.output[1].shape.as_list())

        self.assertEqual([(None, 2048)], model.ae.get_layer('ae_input_vis').input_shape)
        self.assertEqual([(None, 85)], model.ae.get_layer('ae_input_sem').input_shape)
        self.assertEqual((None, 85), model.ae.get_layer('code').output_shape)
        self.assertEqual((None, 85), model.ae.get_layer('ae_output_sem').output_shape)
        self.assertEqual((None, 2048), model.ae.get_layer('ae_output_vis').output_shape)

    def test_zsl_ae(self):
        """
        Tests if zsl autoencoder model is build with the correct size for input, output and encoding
        """
        model = ZSLAutoEncoder(2048 + 85, 85, 2048 + 85)
        model.define_ae()

        self.assertEqual([None, 2048], model.ae.input[0].shape.as_list())
        self.assertEqual([None, 85], model.ae.input[1].shape.as_list())
        self.assertEqual([None, 2048], model.ae.output.shape.as_list())

        self.assertEqual([(None, 2048)], model.ae.get_layer('ae_input_vis').input_shape)
        self.assertEqual([(None, 85)], model.ae.get_layer('ae_input_sem').input_shape)
        self.assertEqual((None, 85), model.ae.get_layer('code').output_shape)
        self.assertEqual((None, 2048), model.ae.get_layer('ae_output_vis').output_shape)

    def test_simple_ae_fitting(self):
        """
        Tests if simple autoencoder model trains as expected
        """
        model = SimpleAutoEncoder(self.train_data.shape[1], self.code_length, self.train_data.shape[1])
        model.fit(self.train_data, self.train_labels, self.test_data, self.test_labels, 2)

        for key in model.history.keys():
            for value in model.history[key]:
                if 'loss' in key:
                    self.assertTrue(value >= 0)
                else:
                    self.assertTrue(0 <= value <= 1)
            self.assertEqual(2, len(model.history[key]))

        self.assertTrue(os.path.isfile('ae_training_history.json'))
        keys = sorted(['svm_train', 'svm_test', 'best_loss', 'loss', 'val_loss'])
        self.assertEqual(keys, sorted(list(model.history.keys())))

    def test_concat_ae_fitting(self):
        """
        Tests if concat autoencoder model trains as expected
        """
        model = ConcatAutoEncoder(self.train_data.shape[1], self.code_length, self.train_data.shape[1])
        model.fit(self.train_data, self.train_labels, self.test_data, self.test_labels, 2)

        for key in model.history.keys():
            for value in model.history[key]:
                if 'loss' in key:
                    self.assertTrue(value >= 0)
                else:
                    self.assertTrue(0 <= value <= 1)
            self.assertEqual(2, len(model.history[key]))

        self.assertTrue(os.path.isfile('ae_training_history.json'))
        keys = sorted(['svm_train', 'svm_test', 'best_loss', 'loss', 'val_loss'])
        self.assertTrue(keys, sorted(list(model.history.keys())))

    def test_zsl_ae_fitting(self):
        """
        Tests if concat autoencoder model trains as expected
        """
        model = ZSLAutoEncoder(self.train_data.shape[1], self.code_length, self.train_data.shape[1])
        vis_length = self.train_data.shape[1] - self.code_length
        model.fit(self.train_data[:, :vis_length], self.train_data[:, vis_length:], 2)

        for key in model.history.keys():
            for value in model.history[key]:
                self.assertTrue(value >= 0)
            self.assertEqual(2, len(model.history[key]))

        self.assertTrue(os.path.isfile('ae_training_history.json'))
        keys = sorted(['best_loss', 'loss'])
        self.assertTrue(keys, sorted(list(model.history.keys())))

    def test_simple_ae_predict(self):
        """
        Tests if simple autoencoder model predicts semantic features as expected
        """
        vis_length = self.train_data.shape[1] - self.code_length
        model = SimpleAutoEncoder(self.train_data.shape[1], self.code_length, self.train_data.shape[1])
        model.fit(self.train_data, self.train_labels, self.test_data, self.test_labels, 2)
        tr_est, te_est = model.predict(self.train_data[:, :vis_length], self.train_data[:, vis_length:],
                                       self.test_data[:, :vis_length], self.test_data[:, vis_length:])

        self.assertEqual((self.train_data.shape[0], self.code_length), tr_est.shape)
        self.assertEqual((self.test_data.shape[0], self.code_length), te_est.shape)

    def test_concat_ae_predict(self):
        """
        Tests if concat autoencoder model predicts semantic features as expected
        """
        vis_length = self.train_data.shape[1] - self.code_length
        model = ConcatAutoEncoder(self.train_data.shape[1], self.code_length, self.train_data.shape[1])
        model.fit(self.train_data, self.train_labels, self.test_data, self.test_labels, 2)
        tr_est, te_est = model.predict(self.train_data[:, :vis_length], self.train_data[:, vis_length:],
                                       self.test_data[:, :vis_length], self.test_data[:, vis_length:])

        self.assertEqual((self.train_data.shape[0], self.code_length), tr_est.shape)
        self.assertEqual((self.test_data.shape[0], self.code_length), te_est.shape)

    def test_zsl_ae_predict(self):
        """
        Tests if zsl autoencoder model predicts semantic features as expected
        """
        vis_length = self.train_data.shape[1] - self.code_length
        model = ZSLAutoEncoder(self.train_data.shape[1], self.code_length, self.train_data.shape[1])
        model.fit(self.train_data[:, :vis_length], self.train_data[:, vis_length:], 2)
        tr_est, te_est = model.predict(self.train_data[:, :vis_length], self.test_data[:, :vis_length])

        self.assertEqual((self.train_data.shape[0], self.code_length), tr_est.shape)
        self.assertEqual((self.test_data.shape[0], self.code_length), te_est.shape)

    @classmethod
    def tearDownClass(cls):
        """
        Deletes files that were written by the tests
        """
        if os.path.isfile('ae_training_history.json'):
            os.remove('ae_training_history.json')
