import os
import logging
import numpy as np
from enum import Enum
from scipy.io import loadmat

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.manifold import Isomap, LocallyLinearEmbedding

from encoders.tools.src.utils import ZSL
from encoders.vse.src.encoder import Encoder, ModelType
from encoders.tools.src.sem_degradation import SemanticDegradation


class DataType(Enum):
    CUB = "CUB"
    AWA = "AWA"
    SUN = "SUN"
    APY = "APY"


class SVMClassifier:
    def __init__(self, data_type, folds, epochs, save=False, results_path='.', degradation_rate=0.0, run_svm=True):
        if type(data_type) != DataType:
            raise ValueError("Invalid data type.")

        self.data_type = data_type
        if self.data_type == DataType.AWA:
            self.lambda_ = 500000
        elif self.data_type == DataType.CUB:
            self.lambda_ = .2
        else:
            self.lambda_ = 0

        self.run_svm = run_svm
        self.n_folds = folds
        self.epochs = epochs
        self.save_results = save
        self.results_path = results_path
        self.degradation_rate = degradation_rate

    def get_te_sem_data(self, data):
        if self.data_type == DataType.AWA:
            lbs = {data['param']['testclasses_id'][0][0][i][0]: attrs for i, attrs in enumerate(data['S_te_pro'])}
            return np.array([lbs[label[0]] for label in data['param']['test_labels'][0][0]])
        elif self.data_type == DataType.CUB:
            lbs = {data['te_cl_id'][i][0]: attributes for i, attributes in enumerate(data['S_te_pro'])}
            return np.array([lbs[label[0]] for label in data['test_labels_cub']])
        else:
            raise ValueError("Invalid data type.")

    def get_data(self, data_path):
        data = loadmat(data_path)

        if self.data_type in (DataType.SUN, DataType.APY):
            sem_data = data['sem_fts'].astype(np.float64)
            vis_data = data['vis_fts'].astype(np.float64)
            lbs_data = np.transpose(data['img_class'])

            return vis_data, lbs_data, sem_data

        vis_data = np.vstack((data['X_tr'], data['X_te']))
        sem_data = np.vstack((data['S_tr'], self.get_te_sem_data(data)))

        if self.data_type == DataType.AWA:
            lbs_data = np.vstack((data['param']['train_labels'][0][0], data['param']['test_labels'][0][0]))
        elif self.data_type == DataType.CUB:
            lbs_data = np.vstack((data['train_labels_cub'], data['test_labels_cub']))
        else:
            raise ValueError("Invalid data type.")

        return vis_data, lbs_data, sem_data

    def classify_vis_data(self, vis_data, labels, reduce_dim=False):
        fold = 0
        accuracies = []
        skf = StratifiedKFold(n_splits=self.n_folds, random_state=None, shuffle=True)

        for train_index, test_index in skf.split(vis_data, labels):
            logging.info('Running VIS classification for fold %d' % fold)

            tr_data, te_data = vis_data[train_index], vis_data[test_index]
            tr_labels, te_labels = labels[train_index][:, 0], labels[test_index][:, 0]

            if reduce_dim:
                tr_data, te_data = ZSL.dimension_reduction(tr_data, te_data, list(tr_labels))

            clf = make_pipeline(StandardScaler(), SVC(gamma='auto', C=1.0, kernel='linear'))
            clf.fit(tr_data, tr_labels)
            prediction = clf.predict(te_data)

            fold += 1
            accuracies.append(balanced_accuracy_score(te_labels, prediction))

        return accuracies

    def classify_sem_data(self, sem_data, labels):
        fold = 0
        accuracies = []
        skf = StratifiedKFold(n_splits=self.n_folds, random_state=None, shuffle=True)

        for train_index, test_index in skf.split(sem_data, labels):
            logging.info('Running SEM classification for fold %d' % fold)

            tr_data = normalize(sem_data[train_index], norm='l2', axis=1, copy=True)

            te_data = normalize(sem_data[test_index], norm='l2', axis=1, copy=True)
            te_data = SemanticDegradation.kill_semantic_attributes(te_data, self.degradation_rate)
            te_data = normalize(te_data, norm='l2', axis=1, copy=True)

            tr_labels, te_labels = labels[train_index][:, 0], labels[test_index][:, 0]

            clf = make_pipeline(StandardScaler(), SVC(gamma='auto', C=1.0, kernel='linear'))
            clf.fit(tr_data, tr_labels)
            prediction = clf.predict(te_data)

            fold += 1
            accuracies.append(balanced_accuracy_score(te_labels, prediction))

        return accuracies

    def classify_concat_data(self, vis_data, sem_data, labels):
        fold = 0
        accuracies = []
        skf = StratifiedKFold(n_splits=self.n_folds, random_state=None, shuffle=True)

        for train_index, test_index in skf.split(vis_data, labels):
            logging.info('Running CAT classification for fold %d' % fold)

            tr_vis, te_vis = vis_data[train_index], vis_data[test_index]
            tr_sem = normalize(sem_data[train_index], norm='l2', axis=1, copy=True)

            te_sem = normalize(sem_data[test_index], norm='l2', axis=1, copy=True)
            te_sem = SemanticDegradation.kill_semantic_attributes(te_sem, self.degradation_rate)
            te_sem = normalize(te_sem, norm='l2', axis=1, copy=True)

            tr_data, te_data = np.hstack((tr_vis, tr_sem)), np.hstack((te_vis, te_sem))
            tr_labels, te_labels = labels[train_index][:, 0], labels[test_index][:, 0]

            clf = make_pipeline(StandardScaler(), SVC(gamma='auto', C=1.0, kernel='linear'))
            clf.fit(tr_data, tr_labels)
            prediction = clf.predict(te_data)

            fold += 1
            accuracies.append(balanced_accuracy_score(te_labels, prediction))

        return accuracies

    def classify_concat_pca_data(self, vis_data, sem_data, labels):
        fold = 0
        accuracies = []
        pca = PCA(n_components=sem_data.shape[1])
        skf = StratifiedKFold(n_splits=self.n_folds, random_state=None, shuffle=True)

        for train_index, test_index in skf.split(vis_data, labels):
            logging.info('Running PCA classification for fold %d' % fold)

            tr_vis = normalize(vis_data[train_index], norm='l2', axis=1, copy=True)
            te_vis = normalize(vis_data[test_index], norm='l2', axis=1, copy=True)
            tr_sem = normalize(sem_data[train_index], norm='l2', axis=1, copy=True)

            te_sem = normalize(sem_data[test_index], norm='l2', axis=1, copy=True)
            te_sem = SemanticDegradation.kill_semantic_attributes(te_sem, self.degradation_rate)
            te_sem = normalize(te_sem, norm='l2', axis=1, copy=True)

            tr_data, te_data = np.hstack((tr_vis, tr_sem)), np.hstack((te_vis, te_sem))
            tr_labels, te_labels = labels[train_index][:, 0], labels[test_index][:, 0]

            clf = make_pipeline(StandardScaler(), SVC(gamma='auto', C=1.0, kernel='linear'))

            pca.fit(tr_data)
            clf.fit(pca.transform(tr_data), tr_labels)
            prediction = clf.predict(pca.transform(te_data))

            fold += 1
            accuracies.append(balanced_accuracy_score(te_labels, prediction))

        return accuracies

    def classify_concat_isomap_data(self, vis_data, sem_data, labels):
        fold = 0
        accuracies = []
        iso = Isomap(n_components=sem_data.shape[1], n_neighbors=20, eigen_solver='auto')
        skf = StratifiedKFold(n_splits=self.n_folds, random_state=None, shuffle=True)

        for train_index, test_index in skf.split(vis_data, labels):
            logging.info('Running ISO classification for fold %d' % fold)

            tr_vis = normalize(vis_data[train_index], norm='l2', axis=1, copy=True)
            te_vis = normalize(vis_data[test_index], norm='l2', axis=1, copy=True)
            tr_sem = normalize(sem_data[train_index], norm='l2', axis=1, copy=True)

            te_sem = normalize(sem_data[test_index], norm='l2', axis=1, copy=True)
            te_sem = SemanticDegradation.kill_semantic_attributes(te_sem, self.degradation_rate)
            te_sem = normalize(te_sem, norm='l2', axis=1, copy=True)

            tr_data, te_data = np.hstack((tr_vis, tr_sem)), np.hstack((te_vis, te_sem))
            tr_labels, te_labels = labels[train_index][:, 0], labels[test_index][:, 0]

            clf = make_pipeline(StandardScaler(), SVC(gamma='auto', C=1.0, kernel='linear'))

            iso.fit(tr_data)
            clf.fit(iso.transform(tr_data), tr_labels)
            prediction = clf.predict(iso.transform(te_data))

            fold += 1
            accuracies.append(balanced_accuracy_score(te_labels, prediction))

        return accuracies

    def classify_concat_lle_data(self, vis_data, sem_data, labels):
        fold = 0
        accuracies = []
        lle = LocallyLinearEmbedding(n_components=sem_data.shape[1], n_neighbors=20)
        skf = StratifiedKFold(n_splits=self.n_folds, random_state=None, shuffle=True)

        for train_index, test_index in skf.split(vis_data, labels):
            logging.info('Running LLE classification for fold %d' % fold)

            tr_vis = normalize(vis_data[train_index], norm='l2', axis=1, copy=True)
            te_vis = normalize(vis_data[test_index], norm='l2', axis=1, copy=True)
            tr_sem = normalize(sem_data[train_index], norm='l2', axis=1, copy=True)

            te_sem = normalize(sem_data[test_index], norm='l2', axis=1, copy=True)
            te_sem = SemanticDegradation.kill_semantic_attributes(te_sem, self.degradation_rate)
            te_sem = normalize(te_sem, norm='l2', axis=1, copy=True)

            tr_data, te_data = np.hstack((tr_vis, tr_sem)), np.hstack((te_vis, te_sem))
            tr_labels, te_labels = labels[train_index][:, 0], labels[test_index][:, 0]

            clf = make_pipeline(StandardScaler(), SVC(gamma='auto', C=1.0, kernel='linear'))

            lle.fit(tr_data)
            clf.fit(lle.transform(tr_data), tr_labels)
            prediction = clf.predict(lle.transform(te_data))

            fold += 1
            accuracies.append(balanced_accuracy_score(te_labels, prediction))

        return accuracies

    def estimate_sae_data(self, tr_vis_data, te_vis_data, tr_sem_data, tr_labels):
        if self.data_type == DataType.CUB:
            tr_vis, te_vis = ZSL.dimension_reduction(tr_vis_data, te_vis_data, tr_labels)
            tr_sem = normalize(tr_sem_data, norm='l2', axis=1, copy=True)

            sae_w = ZSL.sae(tr_vis.transpose(), tr_sem.transpose(), self.lambda_).transpose()
            tr_sem, te_sem = tr_vis.dot(sae_w), te_vis.dot(sae_w)
        else:
            tr_vis = normalize(tr_vis_data.transpose(), norm='l2', axis=1, copy=True).transpose()
            sae_w = ZSL.sae(tr_vis.transpose(), tr_sem_data.transpose(), self.lambda_)

            tr_sem = tr_vis.dot(normalize(sae_w, norm='l2', axis=1, copy=True).transpose())
            te_sem = te_vis_data.dot(normalize(sae_w, norm='l2', axis=1, copy=True).transpose())

        return tr_sem, te_sem

    def classify_sae_data(self, vis_data, sem_data, labels):
        fold = 0
        accuracies = []
        skf = StratifiedKFold(n_splits=self.n_folds, random_state=None, shuffle=True)

        for tr_idx, te_idx in skf.split(vis_data, labels):
            logging.info('Running SAE classification for fold %d' % fold)

            tr_labels, te_labels = labels[tr_idx][:, 0], labels[te_idx][:, 0]

            tr_sem, te_sem = self.estimate_sae_data(vis_data[tr_idx], vis_data[te_idx], sem_data[tr_idx], tr_labels)

            clf = make_pipeline(StandardScaler(), SVC(gamma='auto', C=1.0, kernel='linear'))
            clf.fit(tr_sem, tr_labels)
            prediction = clf.predict(te_sem)

            fold += 1
            accuracies.append(balanced_accuracy_score(te_labels, prediction))

        return accuracies

    def estimate_vse_data(self, tr_vis, te_vis, tr_sem, te_sem, y_train, y_test, res_path):
        tr_sem = normalize(tr_sem, norm='l2', axis=1, copy=True)
        tr_vis = normalize(tr_vis, norm='l2', axis=1, copy=True)
        te_vis = normalize(te_vis, norm='l2', axis=1, copy=True)

        te_sem = normalize(te_sem, norm='l2', axis=1, copy=True)
        te_sem = SemanticDegradation.kill_semantic_attributes(te_sem, self.degradation_rate)
        te_sem = normalize(te_sem, norm='l2', axis=1, copy=True)

        input_length = output_length = tr_vis.shape[1] + tr_sem.shape[1]
        ae = Encoder(input_length, tr_sem.shape[1], output_length, ModelType.STRAIGHT_AE, self.epochs, res_path, self.run_svm)

        tr_sem, te_sem = ae.estimate_semantic_data(tr_vis, te_vis, tr_sem, te_sem, y_train, y_test, self.save_results)

        return tr_sem, te_sem

    def classify_vse_data(self, vis_data, sem_data, labels):
        fold = 0
        accuracies = []
        skf = StratifiedKFold(n_splits=self.n_folds, random_state=None, shuffle=True)

        results_path = os.path.join(self.results_path, 'vse')
        if self.save_results and results_path != '.' and not os.path.isdir(results_path):
            os.mkdir(results_path)

        for tr_idx, te_idx in skf.split(vis_data, labels):
            logging.info('Running VSE classification for fold %d' % fold)

            tr_labels, te_labels = labels[tr_idx][:, 0], labels[te_idx][:, 0]
            res_path = os.path.join(results_path, 'f' + str(fold).zfill(3))

            tr_sem, te_sem = self.estimate_vse_data(vis_data[tr_idx], vis_data[te_idx], sem_data[tr_idx],
                                                    sem_data[te_idx], tr_labels, te_labels, res_path)

            clf = make_pipeline(StandardScaler(), SVC(gamma='auto', C=1.0, kernel='linear'))
            clf.fit(tr_sem, tr_labels)
            prediction = clf.predict(te_sem)

            fold += 1
            accuracies.append(balanced_accuracy_score(te_labels, prediction))

        return accuracies

    def classify_sae2vse_data(self, vis_data, sem_data, labels):
        fold = 0
        accuracies = []
        skf = StratifiedKFold(n_splits=self.n_folds, random_state=None, shuffle=True)

        results_path = os.path.join(self.results_path, 's2s')
        if self.save_results and results_path != '.' and not os.path.isdir(results_path):
            os.mkdir(results_path)

        for tr_idx, te_idx in skf.split(vis_data, labels):
            logging.info('Running S2S classification for fold %d' % fold)

            tr_vis, te_vis = vis_data[tr_idx], vis_data[te_idx]
            tr_labels, te_labels = labels[tr_idx][:, 0], labels[te_idx][:, 0]

            res_path = os.path.join(results_path, 'f' + str(fold).zfill(3))

            tr_sem, te_sem = self.estimate_sae_data(tr_vis, te_vis, sem_data[tr_idx], tr_labels)
            tr_sem, te_sem = self.estimate_vse_data(tr_vis, te_vis, tr_sem, te_sem, tr_labels, te_labels, res_path)

            clf = make_pipeline(StandardScaler(), SVC(gamma='auto', C=1.0, kernel='linear'))
            clf.fit(tr_sem, tr_labels)
            prediction = clf.predict(te_sem)

            fold += 1
            accuracies.append(balanced_accuracy_score(te_labels, prediction))

        return accuracies
