from encoders.tools.src.sem_analysis import SemanticDegradation

sem = SemanticDegradation('../../../../Datasets/SAE/awa_demo_data.mat', data_type='awa', ae_type='se', epochs=50)
sem.write2json(sem.degrade_semantic_data_zsl(n_folds=10), 'awa_v2s_projection_random.json')
sem.write2json(sem.degrade_semantic_data_svm(n_folds=10), 'awa_svm_classification_random.json')

sem = SemanticDegradation('../../../../Datasets/SAE/cub_demo_data.mat', data_type='cub', ae_type='se', epochs=50)
sem.write2json(sem.degrade_semantic_data_zsl(n_folds=10), 'cub_v2s_projection_random.json')
sem.write2json(sem.degrade_semantic_data_svm(n_folds=10), 'cub_svm_classification_random.json')