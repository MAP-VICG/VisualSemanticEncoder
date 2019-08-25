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
from tensorflow.compat.v1.keras.backend import set_session

from core.vsclassifier import SVMClassifier
from core.vsencoder import SemanticEncoder
from core.featuresparser import FeaturesParser
from utils.logwriter import Logger, MessageType


def main():
    init_time = time.time()
    
    mock = True
    
    epochs = 50
    enc_dim = 128
    simple = False
    batch_norm = False
    
    if mock:
        log = Logger(console=True)
        parser = FeaturesParser(fts_dir=os.path.join('features', 'mock'))
        epochs = 5
    else:
        log = Logger(console=False)
        parser = FeaturesParser()
    
    log.write_message('Mock %s' % str(mock), MessageType.INF)
    log.write_message('Simple %s' % str(simple), MessageType.INF)
    log.write_message('Batch Norm %s' % str(batch_norm), MessageType.INF)
    
    sem_fts = parser.get_semantic_features()
    sem_fts = np.multiply(sem_fts, np.array([v for v in range(1, sem_fts.shape[1] + 1)]))
    
    Y = parser.get_labels()
    X = parser.concatenate_features(parser.get_visual_features(), sem_fts)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, stratify=Y, random_state=42, test_size=0.2)
    
    if not mock:
        with open('test_set.txt', 'w') as f:
            for row in x_test:
                f.write(', '.join(map(str, list(row))) + '\n')
        
        with open('test_labels.txt', 'w') as f:
            f.write(', '.join(map(str, list(y_test))))
    
    log.write_message('Starting Semantic Encoder Application', MessageType.INF)
    log.write_message('Autoencoder encoding dimension is %d' % enc_dim, MessageType.INF)
    log.write_message('The model will be trained for %d epochs' % epochs, MessageType.INF)
    
    results = dict()
    
    # classifying visual features
    svm = SVMClassifier()
    log.write_message('Running REF', MessageType.INF)
    results['REF'] = svm.run_svm(x_train=x_train[:,:2048], x_test=x_test[:,:2048], 
                                 y_train=y_train, y_test=y_test, njobs=-1)
           
    # ALL
    enc = SemanticEncoder(epochs, enc_dim)
    log.write_message('Running ALL', MessageType.INF)
    results['ALL'] = enc.run_encoder('ALL', simple, batch_norm,
                                     x_train=enc.pick_semantic_features('ALL', x_train, opposite=False), 
                                     x_test=enc.pick_semantic_features('ALL', x_test, opposite=False), 
                                     y_train=y_train, y_test=y_test)
     
    # COLOR
    log.write_message('Running COLOR', MessageType.INF)
    results['COLOR'] = enc.run_encoder('COLOR', simple, batch_norm,
                                        x_train=enc.pick_semantic_features('COLOR', x_train, opposite=False), 
                                        x_test=enc.pick_semantic_features('COLOR', x_test, opposite=False),
                                        y_train=y_train, y_test=y_test)
      
    # NOT COLOR
    log.write_message('Running NOT COLOR', MessageType.INF)
    results['_COLOR'] = enc.run_encoder('_COLOR', simple, batch_norm,
                                        x_train=enc.pick_semantic_features('COLOR', x_train, opposite=True), 
                                        x_test=enc.pick_semantic_features('COLOR', x_test, opposite=True),
                                        y_train=y_train, y_test=y_test)
      
    # TEXTURE
    log.write_message('Running TEXTURE', MessageType.INF)
    results['TEXTURE'] = enc.run_encoder('TEXTURE', simple, batch_norm,
                                        x_train=enc.pick_semantic_features('TEXTURE', x_train, opposite=False), 
                                        x_test=enc.pick_semantic_features('TEXTURE', x_test, opposite=False),
                                        y_train=y_train, y_test=y_test)
      
    # NOT TEXTURE
    log.write_message('Running NOT TEXTURE', MessageType.INF)
    results['_TEXTURE'] = enc.run_encoder('_TEXTURE', simple, batch_norm,
                                        x_train=enc.pick_semantic_features('TEXTURE', x_train, opposite=True), 
                                        x_test=enc.pick_semantic_features('TEXTURE', x_test, opposite=True),
                                        y_train=y_train, y_test=y_test)
      
    # SHAPE
    log.write_message('Running SHAPE', MessageType.INF)
    results['SHAPE'] = enc.run_encoder('SHAPE', simple, batch_norm,
                                        x_train=enc.pick_semantic_features('SHAPE', x_train, opposite=False), 
                                        x_test=enc.pick_semantic_features('SHAPE', x_test, opposite=False),
                                        y_train=y_train, y_test=y_test)
      
    # NOT SHAPE
    log.write_message('Running NOT SHAPE', MessageType.INF)
    results['_SHAPE'] = enc.run_encoder('_SHAPE', simple, batch_norm,
                                        x_train=enc.pick_semantic_features('SHAPE', x_train, opposite=True), 
                                        x_test=enc.pick_semantic_features('SHAPE', x_test, opposite=True),
                                        y_train=y_train, y_test=y_test)
      
    # PARTS
    log.write_message('Running PARTS', MessageType.INF)
    results['PARTS'] = enc.run_encoder('PARTS', simple, batch_norm,
                                        x_train=enc.pick_semantic_features('PARTS', x_train, opposite=False), 
                                        x_test=enc.pick_semantic_features('PARTS', x_test, opposite=False),
                                        y_train=y_train, y_test=y_test)
      
    # NOT PARTS
    log.write_message('Running NOT PARTS', MessageType.INF)
    results['_PARTS'] = enc.run_encoder('_PARTS', simple, batch_norm,
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
    config = tf.compat.v1.ConfigProto(log_device_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    set_session(tf.compat.v1.Session(config=config))
    
    main()