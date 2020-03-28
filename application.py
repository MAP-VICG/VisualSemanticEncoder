"""
Encodes visual and semantic features of images

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Mar 23, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""
import io
import os
import sys
import time

from sklearn.preprocessing import normalize

from utils.src.configparser import ConfigParser
from utils.src.logwriter import LogWriter, MessageType
from featureextraction.src.dataparsing import DataIO
from encoding.src.encoder import Autoencoder
from encoding.src.plotter import Plotter


def run_ae(x_train, x_test, y_train, y_test, log, config):
    """
    Runs autoencoder training and validation and plots results

    @param x_train: 2D numpy array with train data
    @param x_test: 2D numpy array with test data
    @param y_train: 1D numpy array with train labels
    @param y_test: 1D numpy array with test labels
    @param log: LogWriter object to print messages to log file
    @param config: ConfigParser object with configuration parameters
    """
    # Normalize data
    normalize(x_train, norm='l2', axis=1, copy=False)
    normalize(x_test, norm='l2', axis=1, copy=False)

    # Encode features
    ae = Autoencoder(config.ae_type, x_train.shape[1], config.encoding_size, x_train.shape[1], config.baseline)
    ae.run_ae_model(x_train, y_train, x_test, y_test, config.epochs, njobs=-1)

    # Print results
    log.write_message('SVM Best Accuracy %s' % str(ae.best_accuracy), MessageType.INF)
    log.write_message('SVM Best Parameters %s' % str(ae.svm_best_parameters), MessageType.INF)
    log.write_message('SVM Train Accuracies %s' % str(ae.accuracies['train']), MessageType.INF)
    log.write_message('SVM Test Accuracies %s' % str(ae.accuracies['test']), MessageType.INF)

    try:
        log.write_message('AE Train Accuracies %s' % str(ae.history.history['acc']), MessageType.INF)
        log.write_message('AE Validation Accuracies %s' % str(ae.history.history['val_acc']), MessageType.INF)
        log.write_message('AE Best Accuracy %s' % str(max(ae.history.history['acc'])), MessageType.INF)
    except KeyError:
        pass

    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    ae.model.summary()
    log.write_message('Model summary \n\n%s' % buffer.getvalue(), MessageType.INF)
    sys.stdout = old_stdout

    # Save model
    ae.define_best_models(x_train, y_train, os.path.join(config.results_path, 'ae_weights.h5'))

    # Plot results
    pt = Plotter(ae, config.results_path)
    pt.plot_evaluation(x_test, y_test, config.baseline)


def main():
    init_time = time.time()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--config':
        config_path = sys.argv[2]
    else:
        config_path = os.path.join(os.getcwd(), 'config.xml')
        
    config = ConfigParser(config_path)
    config.read_configuration()
    log = LogWriter(log_path=config.results_path, log_name='semantic_encoder', console=config.console)

    # Print configuration details
    log.write_message('Configuration file: %s' % str(config.configfile), MessageType.INF)
    log.write_message('Data set: n %s' % str(config.dataset), MessageType.INF)
    log.write_message('Results path: %s' % str(config.results_path), MessageType.INF)
    log.write_message('Number of epochs: %s' % str(config.epochs), MessageType.INF)
    log.write_message('Encoding size: %s' % str(config.encoding_size), MessageType.INF)

    log.write_message('x_train path is %s' % config.x_train_path, MessageType.INF)
    log.write_message('y_train path is %s' % config.y_train_path, MessageType.INF)
    log.write_message('x_test path is %s' % config.x_test_path, MessageType.INF)
    log.write_message('y_test path is %s' % config.y_test_path, MessageType.INF)

    log.write_message('Visual features baseline: %f' % config.baseline['vis'], MessageType.INF)

    try:
        # Read features
        x_train = DataIO.get_features(config.x_train_path)
        y_train = DataIO.get_labels(config.y_train_path)

        x_test = DataIO.get_features(config.x_test_path)
        y_test = DataIO.get_labels(config.y_test_path)

        # Run AE
        run_ae(x_train, x_test, y_train, y_test, log, config)

        log.write_message('Execution has finished successfully', MessageType.INF)
    except (IOError, FileNotFoundError) as e:
        log.write_message('Could not read data set. %s' % str(e), MessageType.ERR)

    elapsed = time.time() - init_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    time_elapsed = '{:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds)

    log.write_message('Elapsed time is %s' % time_elapsed, MessageType.INF)
    
    
if __name__ == '__main__':
    main()
