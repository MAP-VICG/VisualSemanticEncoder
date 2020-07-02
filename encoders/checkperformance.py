"""
Computes the encoded space for SAE and SEC for AWA2 and CUB200 data sets. Uses k fold cross validation
to evaluate results. Accuracies are captured for ZSL and SVM classifications.

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Jun 18, 2020
@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""
import os
from encoders.tools.src.sem_analysis import SemanticDegradation

folds = 2
epochs = 3
ae_type = 'sec'
rates = [0, 0.05, 0.1, 0.2, 0.4, 0.8, 0.1]

if not os.path.isdir('results'):
    os.mkdir('results')

if not os.path.isdir(os.path.join('results', 'inception')):
    os.mkdir(os.path.join('results', 'inception'))

if not os.path.isdir(os.path.join('results', 'resnet')):
    os.mkdir(os.path.join('results', 'resnet'))

if not os.path.isdir(os.sep.join(['results', 'inception', ae_type])):
    os.mkdir(os.sep.join(['results', 'inception', ae_type]))

if not os.path.isdir(os.sep.join(['results', 'resnet', ae_type])):
    os.mkdir(os.sep.join(['results', 'resnet', ae_type]))

data = '../../Datasets/SAE/awa_demo_data.mat'
res_path = os.sep.join(['results', 'inception', ae_type])
sem = SemanticDegradation(data, data_type='awa', rates=rates, results_path=res_path)
sem.write2json(sem.degrade_semantic_data(folds, ae_type, epochs), os.path.join(res_path, 'awa_v2s_projection.json'))
sem.write2json(sem.degrade_semantic_data(folds, ae_type, epochs), os.path.join(res_path, 'awa_svm_classification.json'))

data = '../../Datasets/SAE/cub_demo_data.mat'
res_path = os.sep.join(['results', 'inception', ae_type])
sem = SemanticDegradation(data, data_type='cub', rates=rates, results_path=res_path)
sem.write2json(sem.degrade_semantic_data(folds, ae_type, epochs), os.path.join(res_path, 'cub_v2s_projection.json'))
sem.write2json(sem.degrade_semantic_data(folds, ae_type, epochs), os.path.join(res_path, 'cub_svm_classification.json'))

data = '../../Datasets/SAE/awa_demo_data_resnet.mat'
res_path = os.sep.join(['results', 'resnet', ae_type])
sem = SemanticDegradation(data, data_type='awa', rates=rates, results_path=res_path)
sem.write2json(sem.degrade_semantic_data(folds, ae_type, epochs), os.path.join(res_path, 'awa_v2s_projection_resnet.json'))
sem.write2json(sem.degrade_semantic_data(folds, ae_type, epochs), os.path.join(res_path, 'awa_svm_classification_resnet.json'))

data = '../../Datasets/SAE/cub_demo_data_resnet.mat'
res_path = os.sep.join(['results', 'resnet', ae_type])
sem = SemanticDegradation(data, data_type='cub', rates=rates, results_path=res_path)
sem.write2json(sem.degrade_semantic_data(folds, ae_type, epochs), os.path.join(res_path, 'cub_v2s_projection_resent.json'))
sem.write2json(sem.degrade_semantic_data(folds, ae_type, epochs), os.path.join(res_path, 'cub_svm_classification_resnet.json'))
