'''
Model to encode visual and semantic features of images

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Apr 16, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
'''
import os
import gc
import numpy as np
import tensorflow as tf
from xml.dom import minidom
from keras import backend as K
from xml.etree.ElementTree import Element, SubElement, tostring

from core.annotationsparser import AnnotationsParser
from core.vsclassifier import SVMClassifier
from utils.logwriter import Logger, MessageType
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
        
        self.logger = Logger(console=console)
        parser = AnnotationsParser(console=console)
        self.attributes_map = parser.get_attributes_subset(as_dict=True)
        
        self.results_path = os.path.join(os.path.join(os.path.join(os.getcwd().split('SemanticEncoder')[0], 
                                                           'SemanticEncoder'), '_files'), 'results')
        
        if not os.path.isdir(self.results_path):
            os.mkdir(self.results_path)

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
        
    def save_results(self, res_dict):
        '''
        Saves results to XML
        
        @param res_dict: dictionary with results for each category
        @return None
        '''    
        root = Element('SemanticEncoder')
        for key in res_dict.keys():
            if key == 'REF':
                child = SubElement(root, key)
                for k in res_dict[key].keys():
                    sub_child = SubElement(child, k.replace(' ', '_'))
                    for x in res_dict[key][k].keys():
                        sub_sub_child = SubElement(sub_child, x.replace(' ', '_'))
                        sub_sub_child.text = str(res_dict[key][k][x])
            else:
                child = SubElement(root, key)
                for epoch in res_dict[key]:
                    for k in epoch.keys():
                        name = k.replace(' ', '_')
                        sub_child = child.find(name)
                        
                        if not sub_child:
                            sub_child = SubElement(child, name)
                            
                        for x in epoch[k].keys():
                            name = x.replace(' ', '_')
                            sub_sub_child = sub_child.find(name)
                            
                            if sub_sub_child != None:
                                sub_sub_child.text += ',' + str(epoch[k][x])
                            else:
                                sub_sub_child = SubElement(sub_child, name)
                                sub_sub_child.text = str(epoch[k][x])
            
        try:
            result_file = os.path.join(self.results_path, 'ae_results.xml')
            xml = minidom.parseString(tostring(root, 'utf-8')).toprettyxml(indent="  ")
            
            with open(result_file, 'w+') as f:
                f.write(xml)
        
        except (IsADirectoryError, OSError):
            self.logger.write_message('Could not save results under %s.' % result_file, MessageType.ERR)
        
    def __del__(self):
        '''
        Clears memory
        '''
        self.clear_memmory()
