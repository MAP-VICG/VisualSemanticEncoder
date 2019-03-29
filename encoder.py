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
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.backend.tensorflow_backend import set_session

from core.featuresparser import FeaturesParser
from core.annotationsparser import PredicateType
from core.vsautoencoder import VSAutoencoder
from core.vsclassifier import SVMClassifier


ENC_DIM = 32
EPOCHS = 150

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
    x_train, x_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=0.2)
    
    ae = VSAutoencoder(x_train, x_test, y_train, y_test, 5, -1)
    history = ae.run_autoencoder(enc_dim=ENC_DIM, nepochs=EPOCHS, 
                                 results_path=os.path.join(res_path, 'svm_ae_class.txt'))
    
    encoded_fts = ae.encoder.predict(x_test)
    decoded_fts = ae.decoder.predict(encoded_fts)
    
    ae.plot_loss(history.history, os.path.join(res_path, 'ae_loss.png'))
    ae.plot_encoding(x_test, encoded_fts, decoded_fts, os.path.join(res_path, 'ae_encoding.png'))
    ae.plot_pca_vs_encoding(x_test, encoded_fts, os.path.join(res_path, 'ae_components.png'))
    ae.plot_spatial_distribution(x_test, encoded_fts, decoded_fts, y_test, 
                                 os.path.join(res_path, 'ae_distribution.png'))
    clear_memmory()
    
    return ae.svm_history

def run_svm(X, Y, res_path):
    '''
    Runs SVM and saves results
    
    @param X: 2D numpy array with data set
    @param Y: 1D numpy array with labels
    @param res_path: string with path to save results under 
    '''
    x_train, x_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=0.2)
    
    svm = SVMClassifier()
    svm.run_classifier(x_train, y_train, 5, -1)
    
    svm.model.best_estimator_.fit(x_train, y_train)
    pred_dict, prediction = svm.predict(x_test, y_test)
    svm.save_results(prediction, os.path.join(res_path, 'svm_class.txt'))
    
    return pred_dict
    

def plot_classification_results(results_dict, ae_results_dict, results_path):
    '''
    Plots classification results for each model type
    
    @param results_dict: dictionary with classification results
    @param ae_results_dict: dictionary with classification results for autoencoder
    @param results_path: string with full path to save results under
    '''
    try:
        plt.figure()
        plt.rcParams.update({'font.size': 8})
                
        for key in results_dict.keys():
            prediction = [results_dict[key]['weighted avg']['recall'] for _ in range(EPOCHS)]
            plt.plot(prediction, linestyle='dashed')
        
        for key in ae_results_dict.keys():
            learning_curve = [value['weighted avg']['recall'] for value in ae_results_dict[key]]
            plt.plot(learning_curve)
            
        plt.xlabel('Epochs')
        plt.ylabel('Weighted Avg - Recall')
        plt.title('SVM Prediction')
        
        lg = list(results_dict.keys())
        lg.extend(ae_results_dict.keys())
        plt.legend(lg, loc='upper right')
        
        root_path = os.sep.join(results_path.split(os.sep)[:-1])
        if not os.path.isdir(root_path):
            os.mkdir(root_path)
            
        plt.savefig(results_path)
    except OSError:
        print('>> ERROR: SVM plot could not be saved under %s' % results_path)

 
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
    con_fts = parser.concatenate_features(vis_fts, sem_fts)
    
    class_dict = dict()
    ae_class_dict = dict()
    Y = parser.get_labels()
    
    class_dict['vis'] = run_svm(vis_fts, Y, os.path.join(res_path, 'vis'))
    class_dict['sem'] = run_svm(sem_fts, Y, os.path.join(res_path, 'sem'))
    class_dict['con'] = run_svm(con_fts, Y, os.path.join(res_path, 'con'))
    
    ae_class_dict['ae_vis'] = run_encoder(vis_fts, Y, os.path.join(res_path, 'vis'))
    ae_class_dict['ae_sem'] = run_encoder(sem_fts, Y, os.path.join(res_path, 'sem'))
    ae_class_dict['ae_con'] = run_encoder(con_fts, Y, os.path.join(res_path, 'con'))
    
    plot_classification_results(class_dict, ae_class_dict, os.path.join(res_path, 'svm_prediction.png'))

if __name__ == '__main__':
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    set_session(tf.Session(config=config))
    
    main()