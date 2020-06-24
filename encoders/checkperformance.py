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
from encoders.tools.src.sem_analysis import SemanticDegradation

# sem = SemanticDegradation('../../Datasets/SAE/awa_demo_data.mat', data_type='awa', ae_type='sec', epochs=50)
# sem.write2json(sem.degrade_semantic_data_zsl(n_folds=10), 'awa_v2s_projection_random_sec.json')
# sem.write2json(sem.degrade_semantic_data_svm(n_folds=10), 'awa_svm_classification_random_sec.json')
#
# sem = SemanticDegradation('../../Datasets/SAE/awa_demo_resnet_data.mat', data_type='awa', ae_type='sec', epochs=50)
# sem.write2json(sem.degrade_semantic_data_zsl(n_folds=10), 'awa_v2s_projection_random_resnet_sec.json')
# sem.write2json(sem.degrade_semantic_data_svm(n_folds=10), 'awa_svm_classification_random_resnet_sec.json')
#
# sem = SemanticDegradation('../../Datasets/SAE/cub_demo_data.mat', data_type='cub', ae_type='sec', epochs=50)
# sem.write2json(sem.degrade_semantic_data_zsl(n_folds=10), 'cub_v2s_projection_random_sec.json')
# sem.write2json(sem.degrade_semantic_data_svm(n_folds=10), 'cub_svm_classification_random_sec.json')

sem = SemanticDegradation('../../Datasets/SAE/cub_demo_data_resnet.mat', data_type='cub', ae_type='sae', epochs=50)
sem.write2json(sem.degrade_semantic_data_zsl(n_folds=10), 'cub_v2s_projection_random_resent_sec.json')
sem.write2json(sem.degrade_semantic_data_svm(n_folds=2), 'cub_svm_classification_random_resnet_sec.json')
