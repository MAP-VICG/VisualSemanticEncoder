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
import tensorflow as tf
from keras import backend as K

from core.vsclassifier import SVMClassifier
from core.vsautoencoder import VSAutoencoder
from utils.vsplotter import Plotter


class SemanticEncoder:
    def __init__(self, epochs, encoding_dim):
        '''
        Initializes common parameters
        
        @param encoding_dim: autoencoder encoding size
        @param epochs: number of epochs
        '''
        self.epochs = epochs
        self.enc_dim = encoding_dim
        
        self.plotter = Plotter()
        self.svm = SVMClassifier()

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
        
        history = ae.run_autoencoder(enc_dim=min(self.enc_dim, x_train.shape[1]), nepochs=self.epochs)
        
        encoded_fts = ae.encoder.predict(x_test)
        decoded_fts = ae.decoder.predict(encoded_fts)
        
        self.plotter.plot_loss(history.history, tag)
        self.plotter.plot_encoding(x_test, encoded_fts, decoded_fts, tag)
        self.plotter.plot_pca_vs_encoding(x_test, encoded_fts, tag)
        self.plotter.plot_spatial_distribution(x_test, encoded_fts, decoded_fts, y_test, tag)
        
        self.clear_memmory()
        return ae.svm_history
 
    def __del__(self):
        '''
        Clears memory
        '''
        self.clear_memmory()
