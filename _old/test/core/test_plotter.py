"""
Tests for module plotter

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Jun 9, 2019
@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""
import os
import unittest
from sklearn.model_selection import train_test_split

from _old.src.core.plotter import Plotter
from _old.src.core.autoencoder import Autoencoder
from utils.src.configparser import ConfigParser
from _old.src.parser.featuresparser import FeaturesParser


class AutoencoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Initializes model for all tests
        """
        configfile = os.sep.join([os.getcwd().split('test')[0], '_files', 'mockfiles', 'configfiles', 'config.xml'])
        
        config = ConfigParser(configfile)
        config.read_configuration()
        parser = FeaturesParser(fts_dir=config.features_path, console=config.console)
        
        cls.plotter = Plotter(console=True, enc_dim=128)
        
        X = parser.concatenate_features(parser.get_visual_features(), parser.get_semantic_features())
        Y = parser.get_labels()
        
        x_train, cls.x_test, y_train, cls.y_test = train_test_split(X, Y, stratify=Y, test_size=0.2)
        
        cls.enc_dim = 128
        cls.nexamples = x_train.shape[0]
        
        cls.ae = Autoencoder(cv=2, njobs=2, x_train=x_train, y_train=y_train, x_test=cls.x_test, y_test=cls.y_test)
        cls.history = cls.ae.run_autoencoder(cls.enc_dim, 5, 0.1)
         
    def test_plot_evaluation(self):
        """
        Tests if evaluation charts are plot and saved to ae_evaluation.png
        """
        file_name = os.path.join(self.plotter.results_path, 'ae_evaluation.png')
             
        if os.path.isfile(file_name):
            os.remove(file_name)
            
        encoded_fts = self.ae.encoder.predict(self.x_test)
             
        self.plotter.plot_evaluation(self.history.history, self.ae.svm_history, encoded_fts)
        self.assertTrue(os.path.isfile(file_name))
          
    def test_plot_encoding(self):
        """
        Tests if encoding results is plot to ae_encoding.png
        """
        file_name = os.path.join(self.plotter.results_path, 'ae_encoding.png')
              
        if os.path.isfile(file_name):
            os.remove(file_name)
           
        encoded_fts = self.ae.encoder.predict(self.x_test)
        decoded_fts = self.ae.decoder.predict(encoded_fts)
              
        self.plotter.plot_encoding(self.x_test, encoded_fts, decoded_fts)
        self.assertTrue(os.path.isfile(file_name))
           
    def test_plot_spatial_distribution(self):
        """
        Tests if LDA, TSNE and PCA results are plot to ae_distribution.png
        """
        file_name = os.path.join(self.plotter.results_path, 'ae_distribution.png')
              
        if os.path.isfile(file_name):
            os.remove(file_name)
               
        encoded_fts = self.ae.encoder.predict(self.x_test)
        decoded_fts = self.ae.decoder.predict(encoded_fts)
              
        self.plotter.plot_spatial_distribution(self.x_test, encoded_fts, decoded_fts, self.y_test)
        self.assertTrue(os.path.isfile(file_name))
           
    def test_plot_pca_vs_encoding(self):
        """
        Tests if PCA components and encoding components are plot to ae_components.png
        """
        file_name = os.path.join(self.plotter.results_path, 'ae_components.png')
              
        if os.path.isfile(file_name):
            os.remove(file_name)
               
        encoded_fts = self.ae.encoder.predict(self.x_test)
              
        self.plotter.plot_pca_vs_encoding(self.x_test, encoded_fts)
        self.assertTrue(os.path.isfile(file_name))
