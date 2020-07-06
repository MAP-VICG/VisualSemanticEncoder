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

# sem = SemanticDegradation('../../Datasets/SAE/awa_demo_data.mat', rates=[0], results_path='.')
# print(sem.svm_classification_awa(5, 'zsl' 5, 0))

# [0.961074122590928, 0.9528468494492307, 0.9502535421633189, 0.9601449927249226, 0.9352569069808934] - SEC GL
# [0.8464707287354781, 0.8444566065213835, 0.8417966342639481, 0.8431562225010016, 0.8435012531869772]- SAE GL
# [0.8415848933367385, 0.841128530049177, 0.8345170708458616, 0.8467331032874506, 0.8440052908327144] - SAE RN

# sem = SemanticDegradation('../../Datasets/SAE/cub_demo_data.mat', rates=[0], results_path='.')
# print(sem.svm_classification_cub(5, 'zsl' 5, 0))

# [0.8487301218131713, 0.7033852629119248, 0.5877444856196541, 0.8897971668209672, 0.6708411452871129] - SEC GL
# [0.6253357238659414, 0.6234994643275606, 0.6350824341491792, 0.6208262635125937, 0.6226655033360371] - SAE GL
# [0.7247990502370123, 0.7123893343818670, 0.7200052321008096, 0.7122478564106880, 0.6923038974928375] - SAE RN

# sem = SemanticDegradation('../../Datasets/SAE/awa_demo_data.mat', rates=[0], results_path='.')
# print(sem.zsl_classification_awa(5, 'sec', 5, 0))

# [0.8467637540453075, 0.8467637540453075, 0.8467637540453075, 0.8467637540453075, 0.8467637540453075] - SAE GL
# [0.7049391553328561, 0.7049391553328561, 0.7049391553328561, 0.7049391553328561, 0.7049391553328561] - SAE RN
# [0.1384395132426628, 0.1302791696492483, 0.1251252684323550, 0.1272727272727272, 0.0156048675733715] - SEC RN
# [0.1601941747572815, 0.0613268608414239, 0.1344660194174757, 0.0957928802588996, 0.0734627831715210] - SEC GL

# sem = SemanticDegradation('../../Datasets/SAE/cub_demo_data.mat', rates=[0], results_path='.')
# print(sem.zsl_classification_cub(5, 'sec', 5, 0))

# [0.6140470508012275, 0.6140470508012275, 0.6140470508012275, 0.6140470508012275, 0.6140470508012275] - SAE GL
# [0.5584725536992841, 0.5584725536992841, 0.5584725536992841, 0.5584725536992841, 0.5584725536992841] - SAE RN
# [0.0, 0.0, 0.040913740197749744, 0.007841800204568702, 0.05932492328673713] - SEC GL
# [0.020456870098874872, 0.0, 0.0405727923627685, 0.002045687009887487, 0.018411183088987385] - SEC RN

folds = 2
epochs = 2
rates = [0, 0.05]#, 0.1, 0.2, 0.4, 0.8, 0.1]

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
