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
import time
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.backend.tensorflow_backend import set_session

from core.vsclassifier import SVMClassifier
from core.vsencoder import SemanticEncoder
from core.featuresparser import FeaturesParser
from utils.logwriter import Logger, MessageType


def main():
    init_time = time.time()
    
    mock = True
    
    seed = 42
    epochs = 5
    enc_dim = 128
    log = Logger(console=True)
    
    if mock:
        parser = FeaturesParser(fts_dir=os.path.join('features', 'mock'))
    else:
        parser = FeaturesParser()
    
    sem_fts = parser.get_semantic_features()
    sem_fts = np.multiply(sem_fts, np.array([v / 10 for v in range(1, sem_fts.shape[1] + 1)]))
    
    Y = parser.get_labels()
    X = parser.concatenate_features(parser.get_visual_features(), sem_fts)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, stratify=Y, random_state=seed, test_size=0.2)
    
    log.write_message('Starting Semantic Encoder Application', MessageType.INF)
    log.write_message('Autoencoder encoding dimension is %d' % enc_dim, MessageType.INF)
    log.write_message('The model will be trained for %d epochs' % epochs, MessageType.INF)
    
    results = dict()
    
    # classifying visual features
    svm = SVMClassifier()
    results['ref'] = svm.run_svm(x_train=x_train[:,:2048], x_test=x_test[:,:2048], 
                                 y_train=y_train, y_test=y_test)
        
    # ALL
    enc = SemanticEncoder(epochs, enc_dim)
    results['all'] = enc.run_encoder('all', x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    
    # COLORS
    results['col'] = enc.run_encoder('col', x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    
    # NOT COLORS
    results['_col'] = enc.run_encoder('_col', x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    
    # TEXTURE
    results['tex'] = enc.run_encoder('tex', x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    
    # NOT TEXTURE
    results['_tex'] = enc.run_encoder('_tex', x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    
    # SHAPE
    results['shp'] = enc.run_encoder('shp', x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    
    # NOT SHAPE
    results['_shp'] = enc.run_encoder('_shp', x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    
    # PARTS
    results['prt'] = enc.run_encoder('prt', x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    
    # NOT PARTS
    results['_prt'] = enc.run_encoder('_prt', x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    
    elapsed = time.time() - init_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    time_elapsed = '{:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds)
    
    log.write_message('Execution has finished successfully', MessageType.INF)
    log.write_message('Elapsed time is %s' % time_elapsed, MessageType.INF)
    
    
if __name__ == '__main__':
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    set_session(tf.Session(config=config))
    
    main()