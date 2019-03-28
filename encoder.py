'''
Model to encode visual and semantic features of images

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Mar 23, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
'''

import os
import gc
import tensorflow as tf
from keras import backend as K
from keras.utils import normalize
from sklearn.model_selection import train_test_split
from keras.backend.tensorflow_backend import set_session

from core.featuresparser import FeaturesParser
from core.annotationsparser import PredicateType
from core.vsautoencoder import VSAutoencoder

def clear_memmory():
    '''
    Resets Tensorflow graph, clear Keras session and calls garbage collector
    '''
    tf.reset_default_graph()
    K.clear_session()
    gc.collect()

def run_encoder(X, Y, res_path):
    '''
    Runs autoencoder and plots results. It automatically splits the data set into 
    training and test sets
    
    @param X: data set
    @param Y: labels set
    @param res_path: results path to save results under
    @return: dictionary with svm results
    '''
    x_train, x_test, _, y_test = train_test_split(X, Y, stratify=Y, test_size=0.2)
    
    ae = VSAutoencoder()
    history = ae.run_autoencoder(x_train, enc_dim=32, nepochs=150)
    
    encoded_fts = ae.encoder.predict(x_test)
    decoded_fts = ae.decoder.predict(encoded_fts)
    
    ae.plot_loss(history.history, os.path.join(res_path, 'ae_loss.png'))
    ae.plot_encoding(x_test, encoded_fts, decoded_fts, os.path.join(res_path, 'ae_encoding.png'))
    ae.plot_pca_vs_encoding(x_test, encoded_fts, os.path.join(res_path, 'ae_components.png'))
    ae.plot_spatial_distribution(x_test, encoded_fts, decoded_fts, y_test, 
                                 os.path.join(res_path, 'ae_distribution.png'))
    clear_memmory()
    
    return ae.svm_history

def plot_classification_results(results_dict):
    '''
    Plots classification results for each model type
    
    @param results_dict: dictionary with classification results
    '''
    for key in results_dict.keys():
        pass

 
def main():
    fls_path = os.path.join(os.getcwd(), '_files/awa2')
    fts_path = os.path.join(fls_path, 'features/ResNet101')
    res_path = os.path.join(fls_path, 'results')
    ann_path = os.path.join(fls_path, 'base')
    
    if not os.path.isdir(res_path):
        os.mkdir(res_path)
                
    parser = FeaturesParser(fts_path)
    vis_fts = parser.get_visual_features()
    sem_fts = normalize(parser.get_semantic_features(ann_path, 
                                                     PredicateType.CONTINUOUS) + 1, 
                                                     order=1, axis=1)
    class_dict = dict()
    Y = parser.get_labels()
    
    class_dict['vis'] = run_encoder(vis_fts, Y, os.path.join(res_path, 'vis'))
    class_dict['sem'] =run_encoder(sem_fts, Y, os.path.join(res_path, 'sem'))
    class_dict['con'] =run_encoder(parser.concatenate_features(vis_fts, sem_fts), Y, 
                                   os.path.join(res_path, 'con'))
    
    plot_classification_results(class_dict)

if __name__ == '__main__':
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    set_session(tf.Session(config=config))
    
    main()