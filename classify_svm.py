import os
import json
import logging

from encoders.sec.src.encoder import ModelType
from encoders.tools.src.svm_classification import SVMClassifier, DataType


class Classification:
    def __init__(self, folds, epochs, results_path, model_type, save=True):
        self.save = save
        self.folds = folds
        self.epochs = epochs
        self.model_type = model_type
        self.results_path = results_path
        self.result = {'i_cub': dict(), 'r_cub': dict(), 'i_awa': dict(), 'r_awa': dict()}

    def run_classification(self, data_path, label, data_type, rate=0.0):
        rate_label = str(int(rate * 100)).zfill(3)
        results_path = os.sep.join([self.results_path, label, rate_label])

        if self.save and not os.path.isdir(results_path):
            os.makedirs(results_path)

        logging.info('Classifying dataset %s' % label)
        svm = SVMClassifier(data_type, self.model_type, self.folds, self.epochs, degradation_rate=rate)
        vis_data, lbs_data, sem_data = svm.get_data(data_path)

        # self.result[label]['sem'] = svm.classify_sem_data(sem_data, lbs_data)
        # self.result[label]['vis'] = svm.classify_vis_data(vis_data, lbs_data)
        # self.result[label]['sae'] = svm.classify_sae_data(vis_data, sem_data, lbs_data)
        # self.result[label]['cat'] = svm.classify_concat_data(vis_data, sem_data, lbs_data)
        # self.result[label]['pca'] = svm.classify_concat_pca_data(vis_data, sem_data, lbs_data)
        self.result[label]['sec'] = svm.classify_sec_data(vis_data, sem_data, lbs_data, self.save, results_path)
        # self.result[label]['s2s'] = svm.classify_sae2sec_data(vis_data, sem_data, lbs_data, self.save, results_path)

        rate_label = str(int(rate * 100))
        with open(os.path.join(self.results_path, 'classification_results_%s.json' % rate_label.zfill(3)), 'w+') as f:
            json.dump(self.result, f, indent=4, sort_keys=True)

    def classify_all(self, rate):
        logging.info('Running classification for degradation rate of %.2f' % rate)
        # self.run_classification('../Datasets/SEM/cub_demo_data.mat', 'i_cub', DataType.CUB, rate=rate)
        # self.run_classification('../Datasets/SEM/awa_demo_data.mat', 'i_awa', DataType.AWA, rate=rate)
        # self.run_classification('../Datasets/SEM/cub_demo_data_resnet.mat', 'r_cub', DataType.CUB, rate=rate)
        self.run_classification('../Datasets/SEM/awa_demo_data_resnet.mat', 'r_awa', DataType.AWA, rate=rate)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        filename='classify_svm.log',
                        format='%(asctime)s.%(msecs)03d %(levelname)s [%(module)s, %(funcName)s]: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    for degradation_rate in [0.0]:
        klass = Classification(2, 10, 'results_test', ModelType.SIMPLE_AE)
        klass.classify_all(degradation_rate)
