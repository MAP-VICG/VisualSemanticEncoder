"""
Computes PCA for the data set provided and then uses the result as input for Linear SVM classification

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Feb 25, 2020

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""
import time
import tensorflow as tf
import tensorflow.python.keras.backend as K

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score

from utils.src.normalization import Normalization
from utils.src.logwriter import LogWriter, MessageType
from featureextraction.src.dataparsing import DataParser


def main():
    init_time = time.time()

    x_train_path = '../../../Datasets/Birds/features/birds_x_train.txt'
    y_train_path = '../../../Datasets/Birds/features/birds_y_train.txt'
    x_test_path = '../../../Datasets/Birds/features/birds_x_test.txt'
    y_test_path = '../../../Datasets/Birds/features/birds_y_test.txt'

    n_components = 128    # vis+sem 0.023533, vis 0.031191
    # n_components = 256  # vis+sem 0.026071, vis 0.039800
    # n_components = 512  # vis+sem 0.023869, vis 0.036663
    # n_components = 1024 # vis+sem 0.023830, vis 0.036216

    log = LogWriter(log_path='../log', log_name=('pca_%d' % n_components), console=False)
    log.write_message('Computing PCA for data set', MessageType.INF)
    log.write_message('Number of components is %d' % n_components, MessageType.INF)

    log.write_message('x_train path is %s' % x_train_path, MessageType.INF)
    log.write_message('y_train path is %s' % y_train_path, MessageType.INF)
    log.write_message('x_test path is %s' % x_test_path, MessageType.INF)
    log.write_message('y_test path is %s' % y_test_path, MessageType.INF)

    try:
        # Read features
        x_train = DataParser.get_features(x_train_path)
        y_train = DataParser.get_labels(y_train_path)

        x_test = DataParser.get_features(x_test_path)
        y_test = DataParser.get_labels(y_test_path)

        x_train = x_train[:, :2048]
        x_test = x_test[:, :2048]

        log.write_message('x_train shape: %s' % str(x_train.shape), MessageType.INF)
        log.write_message('y_train shape: %s' % str(y_train.shape), MessageType.INF)
        log.write_message('x_test shape: %s' % str(x_test.shape), MessageType.INF)
        log.write_message('y_test shape: %s' % str(x_test.shape), MessageType.INF)

        # Normalize data
        Normalization.normalize_zero_one_by_column(x_train)
        Normalization.normalize_zero_one_by_column(x_test)

        # Compute PCA
        pca = PCA(n_components=n_components)
        training_fts = pca.fit_transform(x_train)
        test_fts = pca.fit_transform(x_test)

        log.write_message('PCA x_train shape: %s' % str(training_fts.shape), MessageType.INF)
        log.write_message('PCA x_test shape: %s' % str(test_fts.shape), MessageType.INF)

        # Apply SVM classification in encoding
        tuning_params = {'kernel': ['linear'], 'C': [0.5, 1, 5, 10]}
        svm = GridSearchCV(SVC(verbose=0, gamma='scale'), tuning_params, cv=5, scoring='recall_macro', n_jobs=-1)
        svm.fit(training_fts, y_train)

        prediction = svm.best_estimator_.predict(test_fts)
        baseline = balanced_accuracy_score(prediction, y_test)

        log.write_message('SVM baseline for PCA is %f' % baseline, MessageType.INF)
        log.write_message('Execution has finished successfully', MessageType.INF)
    except (IOError, FileNotFoundError) as e:
        log.write_message('Could not read data set. %s' % e, MessageType.ERR)
    except ValueError as e:
        log.write_message('There are invalid values in the data. %s' % e, MessageType.ERR)

    elapsed = time.time() - init_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    time_elapsed = '{:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds)

    log.write_message('Elapsed time is %s' % time_elapsed, MessageType.INF)


if __name__ == '__main__':
    config = tf.compat.v1.ConfigProto(log_device_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    K.set_session(tf.compat.v1.Session(config=config))

    main()
