import time
import numpy as np
from scipy.io import loadmat
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.manifold import Isomap, LocallyLinearEmbedding

data = loadmat('../../../Datasets/awa2_data_resnet50.mat')
sem_data = data['sem_fts'].astype(np.float64)
vis_data = data['vis_fts'].astype(np.float64)
labels = np.transpose(data['img_class'])

init_time = time.time()
pca = PCA(n_components=sem_data.shape[1])
lle = LocallyLinearEmbedding(n_components=sem_data.shape[1], n_neighbors=20)
iso = Isomap(n_components=sem_data.shape[1], n_neighbors=20, eigen_solver='auto')
skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)

for train_index, test_index in skf.split(vis_data, labels):
    tr_vis = normalize(vis_data[train_index], norm='l2', axis=1, copy=True)
    te_vis = normalize(vis_data[test_index], norm='l2', axis=1, copy=True)

    tr_sem = normalize(sem_data[train_index], norm='l2', axis=1, copy=True)
    te_sem = normalize(sem_data[test_index], norm='l2', axis=1, copy=True)

    tr_data, te_data = np.hstack((tr_vis, tr_sem)), np.hstack((te_vis, te_sem))
    tr_labels, te_labels = labels[train_index][:, 0], labels[test_index][:, 0]

    clf = make_pipeline(StandardScaler(), SVC(gamma='auto', C=1.0, kernel='linear'))

    pca.fit(tr_data)
    clf.fit(pca.transform(tr_data), tr_labels)
    prediction = clf.predict(pca.transform(te_data))
    print('PCA: %f' % balanced_accuracy_score(te_labels, prediction))

    lle.fit(tr_data)
    clf.fit(lle.transform(tr_data), tr_labels)
    prediction = clf.predict(lle.transform(te_data))
    print('LLE: %f' % balanced_accuracy_score(te_labels, prediction))

    iso.fit(tr_data)
    clf.fit(iso.transform(tr_data), tr_labels)
    prediction = clf.predict(iso.transform(te_data))
    print('ISO: %f' % balanced_accuracy_score(te_labels, prediction))

    break

elapsed = time.time() - init_time
hours, rem = divmod(elapsed, 3600)
minutes, seconds = divmod(rem, 60)
time_elapsed = '{:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds)
print('Elapsed time is %s' % time_elapsed)
