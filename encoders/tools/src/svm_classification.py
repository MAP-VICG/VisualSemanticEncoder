import numpy as np
from enum import Enum
from scipy.io import loadmat

from sklearn.svm import SVC
from encoders.tools.src.utils import ZSL
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score


class DataType(Enum):
    CUB = "CUB"
    AWA = "AWA"


class SVMClassifier:
    def __init__(self, data_type):
        if type(data_type) != DataType:
            raise ValueError("Invalid data type.")

        self.data_type = data_type
        if self.data_type == DataType.AWA:
            self.lambda_ = 500000
        elif self.data_type == DataType.CUB:
            self.lambda_ = .2

    def get_te_sem_data(self, data):
        if self.data_type == DataType.AWA:
            lbs = {data['param']['testclasses_id'][0][0][i][0]: attributes for i, attributes in enumerate(data['S_te_pro'])}
            return np.array([lbs[label[0]] for label in data['param']['test_labels'][0][0]])
        elif self.data_type == DataType.CUB:
            lbs = {data['te_cl_id'][i][0]: attributes for i, attributes in enumerate(data['S_te_pro'])}
            return np.array([lbs[label[0]] for label in data['test_labels_cub']])
        else:
            raise ValueError("Invalid data type.")

    def get_data(self, data_path):
        data = loadmat(data_path)
        vis_data = np.vstack((data['X_tr'], data['X_te']))
        sem_data = np.vstack((data['S_tr'], self.get_te_sem_data(data)))

        if self.data_type == DataType.AWA:
            lbs_data = np.vstack((data['param']['train_labels'][0][0], data['param']['test_labels'][0][0]))
        elif self.data_type == DataType.CUB:
            lbs_data = np.vstack((data['train_labels_cub'], data['test_labels_cub']))
        else:
            raise ValueError("Invalid data type.")

        return vis_data, lbs_data, sem_data

    @staticmethod
    def classify_vis_data(vis_data, labels, n_folds, reduce_dim=False):
        accuracies = []
        skf = StratifiedKFold(n_splits=n_folds, random_state=None, shuffle=True)

        for train_index, test_index in skf.split(vis_data, labels):
            tr_data, te_data = vis_data[train_index], vis_data[test_index]
            tr_labels, te_labels = labels[train_index][:, 0], labels[test_index][:, 0]

            if reduce_dim:
                tr_data, te_data = ZSL.dimension_reduction(tr_data, te_data, list(tr_labels))

            clf = make_pipeline(StandardScaler(), SVC(gamma='auto', C=1.0, kernel='linear'))
            clf.fit(tr_data, tr_labels)
            prediction = clf.predict(te_data)

            accuracies.append(balanced_accuracy_score(te_labels, prediction))

        return accuracies

    @staticmethod
    def classify_sem_data(sem_data, labels, n_folds):
        accuracies = []
        skf = StratifiedKFold(n_splits=n_folds, random_state=None, shuffle=True)

        for train_index, test_index in skf.split(sem_data, labels):
            tr_data = normalize(sem_data[train_index], norm='l2', axis=1)
            te_data = normalize(sem_data[test_index], norm='l2', axis=1)
            tr_labels, te_labels = labels[train_index][:, 0], labels[test_index][:, 0]

            clf = make_pipeline(StandardScaler(), SVC(gamma='auto', C=1.0, kernel='linear'))
            clf.fit(tr_data, tr_labels)
            prediction = clf.predict(te_data)

            accuracies.append(balanced_accuracy_score(te_labels, prediction))

        return accuracies

    @staticmethod
    def classify_concat_data(vis_data, sem_data, labels, n_folds):
        accuracies = []
        skf = StratifiedKFold(n_splits=n_folds, random_state=None, shuffle=True)

        for train_index, test_index in skf.split(vis_data, labels):
            tr_vis, te_vis = vis_data[train_index], vis_data[test_index]
            tr_sem = normalize(sem_data[train_index], norm='l2', axis=1)
            te_sem = normalize(sem_data[test_index], norm='l2', axis=1)

            tr_data, te_data = np.hstack((tr_vis, tr_sem)), np.hstack((te_vis, te_sem))
            tr_labels, te_labels = labels[train_index][:, 0], labels[test_index][:, 0]

            clf = make_pipeline(StandardScaler(), SVC(gamma='auto', C=1.0, kernel='linear'))
            clf.fit(tr_data, tr_labels)
            prediction = clf.predict(te_data)

            accuracies.append(balanced_accuracy_score(te_labels, prediction))

        return accuracies

    def estimate_sae_data(self, tr_vis_data, te_vis_data, tr_sem_data, tr_labels):
        if self.data_type == DataType.CUB:
            tr_vis, te_vis = ZSL.dimension_reduction(tr_vis_data, te_vis_data, tr_labels)
            tr_sem = normalize(tr_sem_data, norm='l2', axis=1)

            sae_w = ZSL.sae(tr_vis.transpose(), tr_sem.transpose(), self.lambda_).transpose()
            tr_sem, te_sem = tr_vis.dot(sae_w), te_vis.dot(sae_w)
        else:
            tr_vis = normalize(tr_vis_data.transpose(), norm='l2', axis=1).transpose()
            sae_w = ZSL.sae(tr_vis.transpose(), tr_sem_data.transpose(), self.lambda_)

            tr_sem = tr_vis.dot(normalize(sae_w, norm='l2', axis=1).transpose())
            te_sem = te_vis_data.dot(normalize(sae_w, norm='l2', axis=1).transpose())

        return tr_sem, te_sem

    def classify_sae_data(self, vis_data, sem_data, labels, n_folds):
        accuracies = []
        skf = StratifiedKFold(n_splits=n_folds, random_state=None, shuffle=True)

        for tr_idx, te_idx in skf.split(vis_data, labels):
            tr_labels, te_labels = labels[tr_idx][:, 0], labels[te_idx][:, 0]

            tr_sem, te_sem = self.estimate_sae_data(vis_data[tr_idx], vis_data[te_idx], sem_data[tr_idx], tr_labels)

            clf = make_pipeline(StandardScaler(), SVC(gamma='auto', C=1.0, kernel='linear'))
            clf.fit(tr_sem, tr_labels)
            prediction = clf.predict(te_sem)

            accuracies.append(balanced_accuracy_score(te_labels, prediction))

        return accuracies
