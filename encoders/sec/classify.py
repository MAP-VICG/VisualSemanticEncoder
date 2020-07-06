import numpy as np
from scipy.io import loadmat
from encoders.sec.src.autoencoder import Autoencoder, ModelType
from sklearn.model_selection import StratifiedKFold

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

from encoders.tools.src.utils import ZSL

data = loadmat('../../../Datasets/SAE/awa_demo_data.mat')
vis_data = np.vstack((data['X_tr'], data['X_te']))
labels = np.vstack((data['param']['train_labels'][0][0], data['param']['test_labels'][0][0]))

labels_dict = {data['param']['testclasses_id'][0][0][i][0]: attributes for i, attributes in enumerate(data['S_te_pro'])}
s_te = np.array([labels_dict[label[0]] for label in data['param']['test_labels'][0][0]])
sem_data = np.vstack((data['S_tr'], s_te))

# data = loadmat('../../../Datasets/SAE/cub_demo_data_resnet.mat')
#
# vis_data = np.vstack((data['X_tr'], data['X_te']))
# labels = np.vstack((data['train_labels_cub'], data['test_labels_cub']))
#
# labels_dict = {data['te_cl_id'][i][0]: attributes for i, attributes in enumerate(data['S_te_pro'])}
# s_te = np.array([labels_dict[label[0]] for label in data['test_labels_cub']])
# sem_data = np.vstack((data['S_tr'], s_te))

fold = accuracy = 0
skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
for train_index, test_index in skf.split(vis_data, labels):
    # tr_vis, te_vis = ZSL.dimension_reduction(vis_data[train_index], vis_data[test_index], list(map(int, labels[train_index])))
    tr_vis, te_vis = vis_data[train_index], vis_data[test_index]

    input_length = output_length = tr_vis.shape[1] + sem_data.shape[1]
    ae = Autoencoder(input_length, sem_data.shape[1], output_length, ModelType.SIMPLE_AE, 5)

    x_train, x_test = ae.estimate_semantic_data(tr_vis, sem_data[train_index], te_vis, sem_data[test_index], labels[train_index])

    clf = make_pipeline(StandardScaler(), SVC(gamma='auto', C=1.0, kernel='linear'))
    clf.fit(x_train, labels[train_index])
    prediction = clf.predict(x_test)
    accuracy = balanced_accuracy_score(prediction, labels[test_index])
    fold += 1

print(fold)
print(accuracy)
print(ae.history['loss'])
print(ae.history['acc'])
print(ae.history['best_loss'])
print(ae.history['best_accuracy'])
