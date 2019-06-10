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
    
    mock = False
    
    epochs = 50
    enc_dim = 128
    log = Logger(console=True)
    
    if mock:
        parser = FeaturesParser(fts_dir=os.path.join('features', 'mock'))
        epochs = 5
    else:
        parser = FeaturesParser()
    
    sem_fts = parser.get_semantic_features()
    sem_fts = np.multiply(sem_fts, np.array([v / 10 for v in range(1, sem_fts.shape[1] + 1)]))
    
    Y = parser.get_labels()
    X = parser.concatenate_features(parser.get_visual_features(), sem_fts)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=0.2)
    
    log.write_message('Starting Semantic Encoder Application', MessageType.INF)
    log.write_message('Autoencoder encoding dimension is %d' % enc_dim, MessageType.INF)
    log.write_message('The model will be trained for %d epochs' % epochs, MessageType.INF)
    
    results = dict()
    
    # classifying visual features
    svm = SVMClassifier()
    results['REF'] = svm.run_svm(x_train=x_train[:,:2048], x_test=x_test[:,:2048], 
                                 y_train=y_train, y_test=y_test)
        
    # ALL
    enc = SemanticEncoder(epochs, enc_dim)
    results['ALL'] = enc.run_encoder('ALL', x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    
    # COLOR
    results['COLOR'] = enc.run_encoder('COLOR', 
                                        x_train=enc.pick_semantic_features('COLOR', x_train, opposite=False), 
                                        x_test=enc.pick_semantic_features('COLOR', x_test, opposite=False),
                                        y_train=y_train, y_test=y_test)
    
    # NOT COLOR
    results['_COLOR'] = enc.run_encoder('_COLOR', 
                                        x_train=enc.pick_semantic_features('COLOR', x_train, opposite=True), 
                                        x_test=enc.pick_semantic_features('COLOR', x_test, opposite=True),
                                        y_train=y_train, y_test=y_test)
    
    # TEXTURE
    results['TEXTURE'] = enc.run_encoder('TEXTURE', 
                                        x_train=enc.pick_semantic_features('TEXTURE', x_train, opposite=False), 
                                        x_test=enc.pick_semantic_features('TEXTURE', x_test, opposite=False),
                                        y_train=y_train, y_test=y_test)
    
    # NOT TEXTURE
    results['_TEXTURE'] = enc.run_encoder('_TEXTURE', 
                                        x_train=enc.pick_semantic_features('TEXTURE', x_train, opposite=True), 
                                        x_test=enc.pick_semantic_features('TEXTURE', x_test, opposite=True),
                                        y_train=y_train, y_test=y_test)
    
    # SHAPE
    results['SHAPE'] = enc.run_encoder('SHAPE', 
                                        x_train=enc.pick_semantic_features('SHAPE', x_train, opposite=False), 
                                        x_test=enc.pick_semantic_features('SHAPE', x_test, opposite=False),
                                        y_train=y_train, y_test=y_test)
    
    # NOT SHAPE
    results['_SHAPE'] = enc.run_encoder('_SHAPE', 
                                        x_train=enc.pick_semantic_features('SHAPE', x_train, opposite=True), 
                                        x_test=enc.pick_semantic_features('SHAPE', x_test, opposite=True),
                                        y_train=y_train, y_test=y_test)
    
    # PARTS
    results['PARTS'] = enc.run_encoder('PARTS', 
                                        x_train=enc.pick_semantic_features('PARTS', x_train, opposite=False), 
                                        x_test=enc.pick_semantic_features('PARTS', x_test, opposite=False),
                                        y_train=y_train, y_test=y_test)
    
    # NOT PARTS
    results['_PARTS'] = enc.run_encoder('_PARTS', 
                                        x_train=enc.pick_semantic_features('PARTS', x_train, opposite=True), 
                                        x_test=enc.pick_semantic_features('PARTS', x_test, opposite=True),
                                        y_train=y_train, y_test=y_test)
    
    enc.save_results(results)
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