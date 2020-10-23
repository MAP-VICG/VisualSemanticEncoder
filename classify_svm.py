import os
import json
import time
import logging
from numba import cuda
import tensorflow as tf

from encoders.tools.src.svm_classification import SVMClassifier, DataType


class Classification:
    def __init__(self, folds, epochs, rate, results_path, save=True):
        self.rate = rate
        self.save = save
        self.folds = folds
        self.epochs = epochs
        self.results_path = results_path
        self.result = {'i_cub': dict(), 'r_cub': dict(), 'i_awa': dict(), 'r_awa': dict(),
                       'r_sun': dict(), 'i_sun': dict(), 'r_apy': dict(), 'i_apy': dict()}

        if self.save and not os.path.isdir(os.path.join(self.results_path)):
            os.makedirs(os.path.join(self.results_path))

        logging.basicConfig(level=logging.DEBUG,
                            filename=os.path.join(results_path, 'classify_svm.log'),
                            format='%(asctime)s %(levelname)s [%(module)s, %(funcName)s]: %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

    def save_results(self, init_time, tag, partial=False):
        rate_label = str(int(self.rate * 100))
        file_name = 'partial_classification_results_%s.json' if partial else 'classification_results_%s.json'

        with open(os.path.join(self.results_path, file_name % rate_label.zfill(3)), 'w+') as f:
            json.dump(self.result, f, indent=4, sort_keys=True)

        elapsed = time.time() - init_time
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        time_elapsed = '{:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds)
        print('Elapsed time %s is %s' % (tag, time_elapsed))

    def run_classification(self, data_path, label, data_type):
        logging.info('Running classification for degradation rate of %.2f' % self.rate)

        rate_label = str(int(self.rate * 100)).zfill(3)
        results_path = os.path.join(self.results_path, label, rate_label)

        if self.save and not os.path.isdir(results_path):
            os.makedirs(results_path)

        logging.info('Classifying dataset %s' % label)
        svm = SVMClassifier(data_type, self.folds, self.epochs, self.save, results_path, self.rate)
        vis_data, lbs_data, sem_data = svm.get_data(data_path)

        init_time = time.time()
        self.result[label]['sem'] = svm.classify_sem_data(sem_data, lbs_data)
        self.save_results(init_time, 'sem', partial=True)

        init_time = time.time()
        self.result[label]['vis'] = svm.classify_vis_data(vis_data, lbs_data)
        self.save_results(init_time, 'vis', partial=True)

        init_time = time.time()
        self.result[label]['sae'] = svm.classify_sae_data(vis_data, sem_data, lbs_data)
        self.save_results(init_time, 'sae', partial=True)

        init_time = time.time()
        self.result[label]['cat'] = svm.classify_concat_data(vis_data, sem_data, lbs_data)
        self.save_results(init_time, 'cat', partial=True)

        init_time = time.time()
        self.result[label]['pca'] = svm.classify_concat_pca_data(vis_data, sem_data, lbs_data)
        self.save_results(init_time, 'pca', partial=True)

        init_time = time.time()
        self.result[label]['iso'] = svm.classify_concat_isomap_data(vis_data, sem_data, lbs_data)
        self.save_results(init_time, 'iso', partial=True)

        init_time = time.time()
        self.result[label]['lle'] = svm.classify_concat_lle_data(vis_data, sem_data, lbs_data)
        self.save_results(init_time, 'lle', partial=True)

        init_time = time.time()
        self.result[label]['vse'] = svm.classify_vse_data(vis_data, sem_data, lbs_data)
        self.save_results(init_time, 'vse', partial=True)

        init_time = time.time()
        self.result[label]['s2s'] = svm.classify_sae2vse_data(vis_data, sem_data, lbs_data)
        self.save_results(init_time, 's2s', partial=False)


if __name__ == '__main__':
    for degradation_rate in [0.0, 0.1, 0.2, 0.3]:
        klass = Classification(5, 50, degradation_rate, 'results_new', save=True)
        # klass.run_classification('../Datasets/awa_data_googlenet.mat', 'i_awa', DataType.AWA)
        # klass.run_classification('../Datasets/awa2_data_resnet50.mat', 'r_awa', DataType.AWA)
        # klass.run_classification('../Datasets/cub_data_googlenet.mat', 'i_cub', DataType.CUB)
        # klass.run_classification('../Datasets/cub_data_resnet50.mat', 'r_cub', DataType.CUB)
        klass.run_classification('../Datasets/apy_data_inceptionv3.mat', 'i_apy', DataType.APY)
        klass.run_classification('../Datasets/apy_data_resnet50.mat', 'r_apy', DataType.APY)
        # klass.run_classification('../Datasets/sun_data_inceptionv3.mat', 'i_sun', DataType.SUN)
        # klass.run_classification('../Datasets/sun_data_resnet50.mat', 'r_sun', DataType.SUN)

        tf.keras.backend.clear_session()
        device = cuda.get_current_device()
        device.reset()
