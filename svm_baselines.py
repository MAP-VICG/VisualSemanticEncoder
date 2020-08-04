import os
import json
from encoders.sec.src.autoencoder import ModelType
from encoders.tools.src.svm_classification import SVMClassifier, DataType


def run_classification(data_path, res_path, label, data_type, model_type=ModelType.SIMPLE_AE, save=True):
    res_path = os.path.join(res_path, label)
    svm = SVMClassifier(data_type, model_type)
    vis_data, lbs_data, sem_data = svm.get_data(data_path)

    if save and not os.path.isdir(res_path):
        os.mkdir(res_path)

    result[label]['sem'] = svm.classify_sem_data(sem_data, lbs_data, folds)
    result[label]['vis'] = svm.classify_vis_data(vis_data, lbs_data, folds)
    result[label]['cat'] = svm.classify_concat_data(vis_data, sem_data, lbs_data, folds)
    result[label]['sae'] = svm.classify_sae_data(vis_data, sem_data, lbs_data, folds)
    result[label]['sec'] = svm.classify_sec_data(vis_data, sem_data, lbs_data, folds, epochs, save, res_path)
    result[label]['s2s'] = svm.classify_sae2sec_data(vis_data, sem_data, lbs_data, folds, epochs, save, res_path)


folds = 5
epochs = 50
result = {'i_cub': dict(), 'r_cub': dict(), 'i_awa': dict(), 'r_awa': dict()}

run_classification('../Datasets/SAE/cub_demo_data.mat', 'results', 'i_cub', DataType.CUB)
run_classification('../Datasets/SAE/cub_demo_data_resnet.mat', 'results', 'r_cub', DataType.CUB)
run_classification('../Datasets/SAE/awa_demo_data.mat', 'results', 'i_awa', DataType.AWA)
run_classification('../Datasets/SAE/awa_demo_data_resnet.mat', 'results', 'r_awa', DataType.AWA)

with open('classification_results.json', 'w+') as f:
    json.dump(result, f, indent=4, sort_keys=True)
