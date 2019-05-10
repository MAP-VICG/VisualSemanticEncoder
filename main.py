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
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from core.encoder import EncodingFeatures
from core.annotationsparser import PredicateType
from utils.logwriter import Logger, MessageType


def main():
    init_time = time.time()
    
    fls_path = os.path.join(os.getcwd(), '_files/awa2')
    fts_path = os.path.join(fls_path, 'features/ResNet101')
    res_path = os.path.join(fls_path, 'results')
    ann_path = os.path.join(fls_path, 'base')
    
    enc_dim = 128
    epochs = 100

    if not os.path.isdir(res_path):
        os.mkdir(res_path)
    
    log = Logger(res_path)
    log.write_message('Starting Semantic Encoder Application', MessageType.INF)
    log.write_message('Autoencoder encoding dimension is %d' % enc_dim, MessageType.INF)
    log.write_message('The model will be trained for %d epochs' % epochs, MessageType.INF)
    
    enc = EncodingFeatures(fts_path, ann_path, res_path, epochs, enc_dim, PredicateType.BINARY)
    
#     enc.encode_visual()
#     enc.encode_semantic()
    enc.encode_concatenated()
    enc.encode_split_features()
    enc.plot_classification_results()
    
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