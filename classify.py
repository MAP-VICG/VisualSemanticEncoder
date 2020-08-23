import os
import json

from encoders.sec.src.autoencoder import ModelType
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
        rate_label = str(int(rate * 100))
        results_path = os.path.join(self.results_path, rate_label)

        if self.save and not os.path.isdir(os.path.join(results_path, label)):
            os.makedirs(os.path.join(results_path, label))

        svm = SVMClassifier(data_type, self.model_type, self.folds, self.epochs, degradation_rate=rate)
        vis_data, lbs_data, sem_data = svm.get_data(data_path)

        self.result[label]['sem'] = svm.classify_sem_data(sem_data, lbs_data)
        self.result[label]['vis'] = svm.classify_vis_data(vis_data, lbs_data)
        self.result[label]['cat'] = svm.classify_concat_data(vis_data, sem_data, lbs_data)
        self.result[label]['sae'] = svm.classify_sae_data(vis_data, sem_data, lbs_data)
        self.result[label]['sec'] = svm.classify_sec_data(vis_data, sem_data, lbs_data, self.save, results_path)
        self.result[label]['s2s'] = svm.classify_sae2sec_data(vis_data, sem_data, lbs_data, self.save, results_path)
        self.result[label]['pca'] = svm.classify_concat_pca_data(vis_data, sem_data, lbs_data)

    def classify_all(self, rate):
        rate_label = str(int(rate * 100))
        self.run_classification('../Datasets/SAE/cub_demo_data.mat', 'i_cub', DataType.CUB, rate=rate)
        self.run_classification('../Datasets/SAE/cub_demo_data_resnet.mat', 'r_cub', DataType.CUB, rate=rate)
        self.run_classification('../Datasets/SAE/awa_demo_data.mat', 'i_awa', DataType.AWA, rate=rate)
        self.run_classification('../Datasets/SAE/awa_demo_data_resnet.mat', 'r_awa', DataType.AWA, rate=rate)

        with open(os.path.join(self.results_path, 'classification_results_%s.json' % rate_label.zfill(3)), 'w+') as f:
            json.dump(self.result, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    for degradation_rate in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        klass = Classification(5, 50, 'results', ModelType.SIMPLE_AE)
        klass.classify_all(degradation_rate)
