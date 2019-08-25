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
    
    mock = False
    
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
    
    classes = [7, 9, 31, 38, 50]
    labels = {7: 'horse', 9: 'blue+whale', 31: 'giraffe', 38: 'zebra', 50: 'dolphin'}
    
    with open('test_set.txt', 'w') as f:
        for i, row in enumerate(x_test):
            if y_test[i] in classes:
                f.write('\t'.join(map(str, list(row))) + '\n')
    
    with open('test_labels.txt', 'w') as f:
        for lb in y_test:
            if lb in classes:
                f.write(labels[lb] + '\n')
                
#         f.write('\n'.join(map(str, list(y_test))))
    
    enc = SemanticEncoder(epochs, enc_dim)
    with open('test_all.txt', 'w') as f:
        for i, row in enumerate(enc.pick_semantic_features('ALL', x_test, opposite=False)):
            if y_test[i] in classes:
                f.write('\t'.join(map(str, list(row))) + '\n')
            
    with open('test_color.txt', 'w') as f:
        for i, row in enumerate(enc.pick_semantic_features('COLOR', x_test, opposite=False)):
            if y_test[i] in classes:
                f.write('\t'.join(map(str, list(row))) + '\n')
            
    with open('test_not_color.txt', 'w') as f:
        for i, row in enumerate(enc.pick_semantic_features('COLOR', x_test, opposite=True)):
            if y_test[i] in classes:
                f.write('\t'.join(map(str, list(row))) + '\n')
            
    with open('test_texture.txt', 'w') as f:
        for i, row in enumerate(enc.pick_semantic_features('TEXTURE', x_test, opposite=False)):
            if y_test[i] in classes:
                f.write('\t'.join(map(str, list(row))) + '\n')
            
    with open('test_not_texture.txt', 'w') as f:
        for i, row in enumerate(enc.pick_semantic_features('TEXTURE', x_test, opposite=True)):
            if y_test[i] in classes:
                f.write('\t'.join(map(str, list(row))) + '\n')
            
    with open('test_shape.txt', 'w') as f:
        for i, row in enumerate(enc.pick_semantic_features('SHAPE', x_test, opposite=True)):
            if y_test[i] in classes:
                f.write('\t'.join(map(str, list(row))) + '\n')
            
    with open('test_not_shape.txt', 'w') as f:
        for i, row in enumerate(enc.pick_semantic_features('SHAPE', x_test, opposite=True)):
            if y_test[i] in classes:
                f.write('\t'.join(map(str, list(row))) + '\n')
            
    with open('test_parts.txt', 'w') as f:
        for i, row in enumerate(enc.pick_semantic_features('PARTS', x_test, opposite=False)):
            if y_test[i] in classes:
                f.write('\t'.join(map(str, list(row))) + '\n')
            
    with open('test_not_parts.txt', 'w') as f:
        for i, row in enumerate(enc.pick_semantic_features('PARTS', x_test, opposite=True)):
            if y_test[i] in classes:
                f.write('\t'.join(map(str, list(row))) + '\n')
            
    log.write_message('Starting Semantic Encoder Application', MessageType.INF)
    log.write_message('Autoencoder encoding dimension is %d' % enc_dim, MessageType.INF)
    log.write_message('The model will be trained for %d epochs' % epochs, MessageType.INF)
    
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