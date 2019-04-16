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
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from core.encoder import EncodingFeatures


def main():
    fls_path = os.path.join(os.getcwd(), 'test/_mockfiles/awa2')
    fts_path = os.path.join(fls_path, 'features/ResNet101')
    res_path = os.path.join(fls_path, 'results')
    ann_path = os.path.join(fls_path, 'base')
    
    enc_dim = 128
    epochs = 30

    if not os.path.isdir(res_path):
        os.mkdir(res_path)
    
    enc = EncodingFeatures(fts_path, ann_path, res_path, epochs, enc_dim)
    enc.encode_visual()
    enc.encode_semantic()
    enc.encode_concatenated()
    enc.plot_classification_results()
    

if __name__ == '__main__':
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    set_session(tf.Session(config=config))
    
    main()