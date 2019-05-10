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
from keras import backend as K
from keras.utils import normalize
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from core.vsclassifier import SVMClassifier
from core.featuresparser import FeaturesParser
from core.annotationsparser import PredicateType
from utils.logwriter import Logger, MessageType
from core.vsautoencoder import VSAutoencoderSingleInput, VSAutoencoderDoubleInput


class SemanticEncoderSingleInput:
    def __init__(self, epochs, encoding_dim, res_path, **kwargs):
        '''
        Initializes common parameters
        
        @param kwargs: dictionary with training and testing data
        @param res_path: string with path to save results under 
        @param encoding_dim: autoencoder encoding size
        @param epochs: number of epochs
        '''
        self.epochs = epochs
        self.enc_dim = encoding_dim
        self.res_path = res_path
        
        self.x_train = kwargs.get('x_train')
        self.y_train = kwargs.get('y_train')
        self.x_test = kwargs.get('x_test')
        self.y_test = kwargs.get('y_test')

    def clear_memmory(self):
        '''
        Resets Tensorflow graph, clear Keras session and calls garbage collector
        '''
        tf.reset_default_graph()
        K.clear_session()
        gc.collect()
    
    def run_encoder(self):
        '''
        Runs autoencoder and plots results. It automatically splits the data set into 
        training and test sets
        
        @return dictionary with svm results
        '''
        ae = VSAutoencoderSingleInput(cv=5, njobs=-1, x_train=self.x_train, x_test=self.x_test, 
                                      y_train=self.y_train, y_test=self.y_test)
        
        history = ae.run_autoencoder(enc_dim=min(self.enc_dim, self.x_train.shape[1]), nepochs=self.epochs, 
                                     results_path=os.path.join(self.res_path, 'svm_ae_class.txt'))
        
        encoded_fts = ae.encoder.predict(self.x_test)
        decoded_fts = ae.decoder.predict(encoded_fts)
        
        ae.plot_loss(history.history, os.path.join(self.res_path, 'ae_loss.png'))
        ae.plot_encoding(self.x_test, encoded_fts, decoded_fts, os.path.join(self.res_path, 'ae_encoding.png'))
        ae.plot_pca_vs_encoding(self.x_test, encoded_fts, os.path.join(self.res_path, 'ae_components.png'))
        ae.plot_spatial_distribution(self.x_test, encoded_fts, decoded_fts, self.y_test, 
                                     os.path.join(self.res_path, 'ae_distribution.png'))
        self.clear_memmory()
        
        return ae.svm_history
    
    def run_svm(self):
        '''
        Runs SVM and saves results
        
        @return dictionary with svm results
        '''
        svm = SVMClassifier()
        svm.run_classifier(self.x_train, self.y_train, 5, -1)
        
        svm.model.best_estimator_.fit(self.x_train, self.y_train)
        pred_dict, prediction = svm.predict(self.x_test, self.y_test)
        svm.save_results(prediction, os.path.join(self.res_path, 'svm_class.txt'))
        
        return pred_dict

    def __del__(self):
        '''
        Clears memory
        '''
        self.clear_memmory()
        

class SemanticEncoderDoubleInput(SemanticEncoderSingleInput):
    def __init__(self, epochs, encoding_dim, res_path, **kwargs):
        '''
        Initializes common parameters
        
        @param kwargs: dictionary with training and testing data
        @param res_path: string with path to save results under 
        @param encoding_dim: autoencoder encoding size
        @param epochs: number of epochs
        '''
        self.epochs = epochs
        self.enc_dim = encoding_dim
        self.res_path = res_path
        
        self.split = kwargs.get('split')
        self.x_train = kwargs.get('x_train')
        self.x_test = kwargs.get('x_test')
        self.y_train = kwargs.get('y_train')
        self.y_test = kwargs.get('y_test')
    
    def run_encoder(self):
        '''
        Runs autoencoder and plots results. It automatically splits the data set into 
        training and test sets
        
        @return dictionary with svm results
        '''
        ae = VSAutoencoderDoubleInput(cv=5, njobs=-1, x_train=self.x_train, x_test=self.x_test, 
                                      y_train=self.y_train, y_test=self.y_test, split=self.split)
        
        history = ae.run_autoencoder(enc_dim=min(self.enc_dim, self.x_train[:,:self.split].shape[1]), nepochs=self.epochs, 
                                     results_path=os.path.join(self.res_path, 'svm_ae_class.txt'))

        encoded_fts = ae.encoder.predict([self.x_test[:,:self.split], np.expand_dims(self.x_test[:,self.split:], axis=-1)])
        decoded_fts = ae.decoder.predict(encoded_fts)
        
        ae.plot_loss(history.history, os.path.join(self.res_path, 'ae_loss.png'))
        ae.plot_encoding(self.x_test, encoded_fts, decoded_fts, os.path.join(self.res_path, 'ae_encoding.png'))
        ae.plot_pca_vs_encoding(self.x_test, encoded_fts, os.path.join(self.res_path, 'ae_components.png'))
        ae.plot_spatial_distribution(self.x_test, encoded_fts, decoded_fts, self.y_test, 
                                     os.path.join(self.res_path, 'ae_distribution.png'))
        self.clear_memmory()
        
        return ae.svm_history
    

class EncodingFeatures:
    def __init__(self, fts_path, ann_path, res_path, epochs, enc_dim, pred_type=PredicateType.CONTINUOUS):
        '''
        Retrieves model input data
        
        @param fts_path: sting with features path
        @param ann_path: string with annotations path
        @param res_path: string with results path
        @param encoding_dim: autoencoder encoding size
        @param epochs: number of epochs
        '''
        parser = FeaturesParser(fts_path)
        self.vis_fts = parser.get_visual_features()
        
        if pred_type == PredicateType.CONTINUOUS:
            self.sem_fts = parser.get_semantic_features(ann_path, PredicateType.CONTINUOUS)
            self.sem_fts = normalize(self.sem_fts + 1, order=1, axis=1)
        else:
            self.sem_fts = parser.get_semantic_features(ann_path, PredicateType.BINARY)
            self.sem_fts = np.multiply(self.sem_fts, np.array([v for v in range(1, self.sem_fts.shape[1] + 1)]))
            
            color_mask = np.zeros((self.sem_fts.shape[1],))
            for k in range(0, 7):
                color_mask[k] = 1
                
            self.sem_fts = np.multiply(self.sem_fts, np.array(color_mask))
    
        self.seed = 42
        self.test_size = 0.2
        self.res_path = res_path
        self.epochs = epochs
        self.enc_dim = enc_dim
        
        self.results_dict = dict()
        self.ae_results_dict = dict()
        self.labels = parser.get_labels()
        
    def encode_visual(self):
        '''
        Runs encoding for visual features and saves results to dictionary
        '''
        Logger().write_message('Encoding visual features...', MessageType.INF)
        
        x_train, x_test, y_train, y_test = train_test_split(self.vis_fts, 
                                                            self.labels, 
                                                            stratify=self.labels, 
                                                            shuffle=True, 
                                                            random_state=self.seed, 
                                                            test_size=self.test_size)
    
        enc = SemanticEncoderSingleInput(self.epochs, self.enc_dim, x_train=x_train, x_test=x_test, y_train=y_train, 
                              y_test=y_test, res_path=os.path.join(self.res_path, 'vis'))
        
        Logger().write_message('Classifying visual features...', MessageType.INF)
        self.results_dict['vis'] = enc.run_svm()
        
        Logger().write_message('Classifying encoded visual features...', MessageType.INF)
        self.ae_results_dict['ae_vis'] = enc.run_encoder()
        
    def encode_semantic(self):
        '''
        Runs encoding for semantic features and saves results to dictionary
        '''
        Logger().write_message('Encoding semantic features...', MessageType.INF)
        
        x_train, x_test, y_train, y_test = train_test_split(self.sem_fts, 
                                                            self.labels, 
                                                            stratify=self.labels, 
                                                            shuffle=True, 
                                                            random_state=self.seed, 
                                                            test_size=self.test_size)
     
        enc = SemanticEncoderSingleInput(self.epochs, self.enc_dim, x_train=x_train, x_test=x_test, y_train=y_train, 
                              y_test=y_test, res_path=os.path.join(self.res_path, 'sem'))
        
        Logger().write_message('Classifying semantic features...', MessageType.INF)
        self.results_dict['sem'] = enc.run_svm()
        
        Logger().write_message('Classifying encoded semantic features...', MessageType.INF)
        self.ae_results_dict['ae_sem'] = enc.run_encoder()
        
    def encode_concatenated(self):
        '''
        Runs encoding for semantic and visual features concatenated and saves 
        results to dictionary
        '''
        Logger().write_message('Encoding concatenated features...', MessageType.INF)
        con_fts = FeaturesParser.concatenate_features(self.vis_fts, self.sem_fts)
        x_train, x_test, y_train, y_test = train_test_split(con_fts, 
                                                            self.labels, 
                                                            stratify=self.labels, 
                                                            shuffle=True, 
                                                            random_state=self.seed, 
                                                            test_size=self.test_size)
    
        enc = SemanticEncoderSingleInput(self.epochs, self.enc_dim, x_train=x_train, x_test=x_test, y_train=y_train, 
                              y_test=y_test, res_path=os.path.join(self.res_path, 'con'))
        
        Logger().write_message('Classifying concatenated features...', MessageType.INF)
        self.results_dict['con'] = enc.run_svm()
        
        Logger().write_message('Classifying encoded concatenated features...', MessageType.INF)
        self.ae_results_dict['ae_con'] = enc.run_encoder()
        
    def encode_split_features(self):
        '''
        Runs encoding for semantic and visual features concatenated and saves 
        results to dictionary
        '''
        Logger().write_message('Encoding split features...', MessageType.INF)
        
        con_fts = FeaturesParser.concatenate_features(self.vis_fts, self.sem_fts)
        x_train, x_test, y_train, y_test = train_test_split(con_fts, 
                                                            self.labels, 
                                                            stratify=self.labels, 
                                                            shuffle=True, 
                                                            random_state=self.seed, 
                                                            test_size=self.test_size)
    
        enc = SemanticEncoderDoubleInput(self.epochs, self.enc_dim, x_train=x_train, x_test=x_test, y_train=y_train, 
                                         y_test=y_test, split=self.vis_fts.shape[1], res_path=os.path.join(self.res_path, 'spt'))
        
#         Logger().write_message('Classifying concatenated features...', MessageType.INF)
#         self.results_dict['spt'] = enc.run_svm()

        Logger().write_message('Classifying encoded split features...', MessageType.INF)
        self.ae_results_dict['ae_spt'] = enc.run_encoder()
    
    def plot_classification_results(self):
        '''
        Plots classification results for each model type
        
        @param results_dict: dictionary with classification results
        @param ae_results_dict: dictionary with classification results for autoencoder
        '''
        try:
            plt.figure()
            plt.rcParams.update({'font.size': 8})
            
            lg = []        
            styles = ['dashed', 'dotted', 'dashdot', 'dotted']
            for i, key in enumerate(self.results_dict.keys()):
                prediction = [self.results_dict[key]['weighted avg']['recall'] for _ in range(self.epochs)]
                plt.plot(prediction, linestyle=styles[i])
                lg.append(key)
            
            for key in self.ae_results_dict.keys():
                learning_curve = [value['weighted avg']['recall'] for value in self.ae_results_dict[key]]
                plt.plot(learning_curve)
                lg.append(key)
                
            plt.xlabel('Epochs')
            plt.ylabel('Weighted Avg - Recall')
            plt.title('SVM Prediction')
            plt.legend(lg, loc='upper right')
            
            res_path = os.path.join(self.res_path, 'svm_prediction.png')
            
            plt.savefig(res_path)
        except OSError:
            Logger().write_message('SVM plot could not be saved under %s.' % res_path, MessageType.ERR)
