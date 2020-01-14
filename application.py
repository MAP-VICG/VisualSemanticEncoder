"""
Encodes visual and semantic features of images

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Mar 23, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""
import os
import sys
import time
import tensorflow as tf
import tensorflow.python.keras.backend as K

from utils.src.configparser import ConfigParser
from utils.src.logwriter import LogWriter, MessageType
from featureextraction.src.dataparsing import DataParser


def main():
    init_time = time.time()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--config':
        config_path = sys.argv[2]
    else:
        config_path = os.sep.join([os.getcwd(), '_files', 'config.xml'])
        
    config = ConfigParser(config_path)
    config.read_configuration()
    
    log = LogWriter(log_path=config.results_path, console=config.console)
    log.write_message('Configuration file %s' % str(config.configfile), MessageType.INF)
    log.write_message('Features path %s' % str(config.features_path), MessageType.INF)
    log.write_message('Results path %s' % str(config.results_path), MessageType.INF)

    log.write_message('Noise rate %s' % str(config.noise_rate), MessageType.INF)
    log.write_message('Autoencoder noise rate %s' % str(config.ae_noise_factor), MessageType.INF)
    log.write_message('Number of epochs %s' % str(config.epochs), MessageType.INF)
    log.write_message('Encoding size %s' % str(config.encoding_size), MessageType.INF)

    # Read features
    try:
        x_train = DataParser.get_features(config.x_train_path)
        y_train = DataParser.get_features(config.y_train_path)

        x_test = DataParser.get_features(config.x_test_path)
        y_test = DataParser.get_features(config.y_test_path)
    except (IOError, FileNotFoundError) as e:
        log.write_message('Could not read data set. %s' % str(e), MessageType.ERR)
        exit(-1)
    except ValueError as e:
        log.write_message('There are invalid values in the data. %s' % str(e), MessageType.ERR)

    # Encode features

    # Save results

    elapsed = time.time() - init_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    time_elapsed = '{:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds)

    log.write_message('Execution has finished successfully', MessageType.INF)
    log.write_message('Elapsed time is %s' % time_elapsed, MessageType.INF)
    
    
if __name__ == '__main__':
    config = tf.compat.v1.ConfigProto(log_device_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    K.set_session(tf.compat.v1.Session(config=config))
    
    main()
