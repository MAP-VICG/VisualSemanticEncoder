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

folds = 5
epochs = 30
rates = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]

if not os.path.isdir('results'):
    os.mkdir('results')

if not os.path.isdir(os.path.join('results', 'inception')):
    os.mkdir(os.path.join('results', 'inception'))

if not os.path.isdir(os.path.join('results', 'resnet')):
    os.mkdir(os.path.join('results', 'resnet'))

data = '../../Datasets/SAE/awa_demo_data.mat'
res_path = os.sep.join(['results', 'inception'])
sem = SemanticDegradation(data, rates=rates, results_path=res_path)
sem.write2json(sem.degrade_semantic_data('awa', 'zsl', folds, epochs), os.path.join(res_path, 'awa_zsl.json'))
sem.write2json(sem.degrade_semantic_data('awa', 'cls', folds, epochs), os.path.join(res_path, 'awa_svm.json'))

data = '../../Datasets/SAE/cub_demo_data.mat'
res_path = os.sep.join(['results', 'inception'])
sem = SemanticDegradation(data, rates=rates, results_path=res_path)
sem.write2json(sem.degrade_semantic_data('cub', 'zsl', folds, epochs), os.path.join(res_path, 'cub_zsl.json'))
sem.write2json(sem.degrade_semantic_data('cub', 'cls', folds, epochs), os.path.join(res_path, 'cub_svm.json'))

data = '../../Datasets/SAE/awa_demo_data_resnet.mat'
res_path = os.sep.join(['results', 'resnet'])
sem = SemanticDegradation(data, rates=rates, results_path=res_path)
sem.write2json(sem.degrade_semantic_data('awa', 'zsl', folds, epochs), os.path.join(res_path, 'awa_zsl.json'))
sem.write2json(sem.degrade_semantic_data('awa', 'cls', folds, epochs), os.path.join(res_path, 'awa_svm.json'))

data = '../../Datasets/SAE/cub_demo_data_resnet.mat'
res_path = os.sep.join(['results', 'resnet'])
sem = SemanticDegradation(data, rates=rates, results_path=res_path)
sem.write2json(sem.degrade_semantic_data('cub', 'zsl', folds, epochs), os.path.join(res_path, 'cub_zsl.json'))
sem.write2json(sem.degrade_semantic_data('cub', 'cls', folds, epochs), os.path.join(res_path, 'cub_svm.json'))
