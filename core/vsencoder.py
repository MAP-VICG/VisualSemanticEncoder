'''
Model to encode visual and semantic features of images

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Apr 16, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
'''
import gc
import numpy as np
import tensorflow as tf
from keras import backend as K

from core.annotationsparser import AnnotationsParser
from core.vsclassifier import SVMClassifier
from core.vsautoencoder import VSAutoencoder
from utils.vsplotter import Plotter


class SemanticEncoder:
    def __init__(self, epochs, encoding_dim, console=False):
        '''
        Initializes common parameters
        
        @param encoding_dim: autoencoder encoding size
        @param epochs: number of epochs
        @param console: if True, prints debug in console
        '''
        self.epochs = epochs
        self.enc_dim = encoding_dim
        
        self.plotter = Plotter()
        self.svm = SVMClassifier()
        
        parser = AnnotationsParser(console=console)
        self.attributes_map = parser.get_attributes_subset(as_dict=True)

    def clear_memmory(self):
        '''
        Resets Tensorflow graph, clear Keras session and calls garbage collector
        '''
        tf.reset_default_graph()
        K.clear_session()
        gc.collect()
    
    def run_encoder(self, tag, **kwargs):
        '''
        Runs autoencoder and plots results. It automatically splits the data set into 
        training and test sets
        
        @param tag: string with folder name to saver results under
        @param kwargs: dictionary with training and testing data
        @return dictionary with svm results
        '''
        x_train = kwargs.get('x_train')
        y_train = kwargs.get('y_train')
        x_test = kwargs.get('x_test')
        y_test = kwargs.get('y_test')
        
        ae = VSAutoencoder(cv=5, njobs=-1, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
        
        history = ae.run_autoencoder(enc_dim=min(self.enc_dim, x_train.shape[1]), nepochs=self.epochs, tag=tag)
        
        encoded_fts = ae.encoder.predict(x_test)
        decoded_fts = ae.decoder.predict(encoded_fts)
        
        self.plotter.plot_loss(history.history, tag)
        self.plotter.plot_encoding(x_test, encoded_fts, decoded_fts, tag)
        self.plotter.plot_pca_vs_encoding(x_test, encoded_fts, tag)
        self.plotter.plot_spatial_distribution(x_test, encoded_fts, decoded_fts, y_test, tag)
        
        self.clear_memmory()
        return ae.svm_history
    
    def pick_semantic_features(self, key, dataset, opposite=False):
        '''
        Builds a data set only with the features indicated by the key. Key is string that determines
        a list of features under a category.
        
        @param key: string with features category
        @param dataset: numpy array with data set to be filtered
        @param opposite: if True instead of getting the features in key, get all features but it
        @return numpy array with filtered features
        '''
        if opposite:
            att_fts = [fts for k in self.attributes_map.keys() for fts in self.attributes_map[k] if k != key]
        else:
            att_fts = self.attributes_map[key]
            
        new_dataset = np.zeros((dataset.shape[0], 2048 + len(att_fts)))
        
        for idx, fts in enumerate(att_fts):
            new_dataset[:, 2048 + idx] = dataset[:, 2048 + fts[0]]
        
        return new_dataset
        
    def __del__(self):
        '''
        Clears memory
        '''
        self.clear_memmory()
