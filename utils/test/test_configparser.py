"""
Tests for module configparser

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Nov 17, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""
import os
import unittest

from utils.src.configparser import ConfigParser
from encoding.src.encoder import ModelType


class ConfigParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Initializes parser for all tests
        """
        configfile = os.sep.join(['mockfiles', 'configfiles', 'config.xml'])
        cls.parser = ConfigParser(configfile)
        cls.parser.read_configuration()
        
    def test_init_parser_invalid_file(self):
        """
        Tests is ValueError exception is raised when XML file is invalid
        """
        self.assertRaises(ValueError, ConfigParser, os.getcwd())

    def test_node_not_found(self):
        """
        Tests if AttributeError exception is raised when node was not found in XML
        """
        configfile = os.sep.join(['mockfiles', 'configfiles', 'config_node_err.xml'])
        parser = ConfigParser(configfile)

        self.assertRaises(AttributeError, parser.read_configuration)

    def test_console_flag(self):
        """
        Tests if the correct console value was found
        """
        self.assertTrue(self.parser.console)

    def test_dataset_name(self):
        """
        Tests if the correct data set value was found
        """
        self.assertEqual('awa2', self.parser.dataset)

    def test_autoencoder_type(self):
        """
        Tests if the correct auto encoder type value was found
        """
        self.assertEqual(ModelType.SIMPLE_AE, self.parser.ae_type)

    def test_baselines(self):
        """
        Tests if the correct baselines values were found
        """
        self.assertEqual({'vis': 0.626, 'stk': 0.745, 'tnn': 0.0, 'pca': 0.031191}, self.parser.baseline)

    def test_num_epochs(self):
        """
        Tests if the correct number of epochs value was found
        """
        self.assertEqual(5, self.parser.epochs)
        
    def test_encoding_size(self):
        """
        Tests if the correct encoding size value was found
        """
        self.assertEqual(128, self.parser.encoding_size)

    def test_output_size(self):
        """
        Tests if the correct output size value was found
        """
        self.assertEqual(2048, self.parser.output_size)

    def test_results_path(self):
        """
        Tests if the correct results_128 path value was found
        """
        self.assertEqual('./_files/results/', self.parser.results_path)

    def test_x_train_path(self):
        """
        Tests if the correct features path value was found
        """
        self.assertEqual('../Datasets/Birds/features/birds_x_train.txt', self.parser.x_train_path)

    def test_y_train_path(self):
        """
        Tests if the correct features path value was found
        """
        self.assertEqual('../Datasets/Birds/features/birds_y_train.txt', self.parser.y_train_path)

    def test_x_test_path(self):
        """
        Tests if the correct features path value was found
        """
        self.assertEqual('../Datasets/Birds/features/birds_x_test.txt', self.parser.x_test_path)

    def test_y_test_path(self):
        """
        Tests if the correct features path value was found
        """
        self.assertEqual('../Datasets/Birds/features/birds_y_test.txt', self.parser.y_test_path)

    def test_chosen_classes(self):
        """
        Tests if the correct chosen_classes value was found
        """
        self.assertEqual([50, 9, 7, 31, 38], self.parser.chosen_classes)

    def test_classes_names(self):
        """
        Tests if the correct classes_names value was found
        """
        self.assertEqual(['dolphin', 'blue+whale', 'horse', 'giraffe', 'zebra'], self.parser.classes_names)
