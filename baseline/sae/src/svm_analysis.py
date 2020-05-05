import json
import random
import numpy as np
from scipy.io import loadmat

from sklearn.svm import SVC
from sklearn.preprocessing import normalize
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

from baseline.sae.src.utils import ZSL


class SVMClassification:
    def __init__(self, data_type):
        """
        Initializes auxiliary variables

        :param data_type: type of data to compute (awa or cub)
        """
        self.data_type = data_type
        self.tuning_params = {'kernel': ['linear'], 'C': [0.5, 1, 5, 10]}

    def estimate_semantic_data(self, vis_tr_data, vis_te_data, sem_tr_data):
        """
        Trains SAE and applies it to visual data in order to estimate its correspondent semantic data

        :param vis_tr_data: visual training data
        :param vis_te_data: visual test data
        :param sem_tr_data: semantic training data
        :return: 2D numpy array with estimated semantic data
        """
        if self.data_type == 'awa':
            x_tr = normalize(vis_tr_data.transpose(), norm='l2', axis=1, copy=True).transpose()
            w = ZSL.sae(x_tr.transpose(), sem_tr_data.transpose(), 500000)
            return vis_te_data.dot(normalize(w, norm='l2', axis=1, copy=True).transpose())
        elif self.data_type == 'cub':
            s_tr = normalize(sem_tr_data, norm='l2', axis=1, copy=False)
            w = ZSL.sae(vis_tr_data.transpose(), s_tr.transpose(), .2).transpose()
            return vis_te_data.dot(w)
        else:
            raise ValueError('Unknown type of data')

    def structure_data(self, data_file):
        """
        Loads data and structures it in a unique set of semantic data, visual data and labels so
        SVM can be applied.

        :param data_file: string with path to .mat data
        :return: tuple with arrays for semantic data, visual data and labels
        """
        data = loadmat(data_file)
        if self.data_type == 'awa':
            y_tr = data['param']['train_labels'][0][0]
            y_te = data['param']['test_labels'][0][0]
            attr_id = data['param']['testclasses_id'][0][0]
            x_tr, x_te = data['X_tr'], data['X_te']
        elif self.data_type == 'cub':
            y_tr = data['train_labels_cub']
            y_te = data['test_labels_cub']
            attr_id = data['te_cl_id']
            x_tr, x_te = ZSL.dimension_reduction(data['X_tr'], data['X_te'], list(map(int, data['train_labels_cub'])))
        else:
            raise ValueError('Unknown type of data')

        labels_dict = {attr_id[i][0]: attributes for i, attributes in enumerate(data['S_te_pro'])}
        s_te = np.array([labels_dict[label[0]] for label in y_te])

        return np.vstack((data['S_tr'], s_te)), np.vstack((x_tr, x_te)), np.vstack((y_tr, y_te))[:, 0]

    def classify_data(self, sem_data, vis_data, labels, n_folds):
        """
        Runs SVM defined by GridSearch for n folds. Training data is defined by the semantic data, and
        test data is estimated through SAE.

        :param sem_data: semantic data
        :param vis_data: visual data
        :param labels: data labels
        :param n_folds: number of folds to split data into and apply classification
        :return: tuple with list of classification accuracies and list of parameters chosen per fold
        """
        accuracies = list()
        best_params = {'kernel': [], 'C': []}
        skf = StratifiedKFold(n_splits=n_folds, random_state=None, shuffle=True)

        for train_index, test_index in skf.split(sem_data, labels):
            x_train = sem_data[train_index]
            x_test = self.estimate_semantic_data(vis_data[train_index], vis_data[test_index], sem_data[train_index])
            y_train, y_test = labels[train_index], labels[test_index]

            svm_model = SVC(verbose=0, max_iter=1000, gamma='scale')
            svm_model = GridSearchCV(svm_model, self.tuning_params, cv=5, scoring='recall_macro', n_jobs=-1)

            svm_model.fit(x_train, y_train)
            best_params['kernel'].append(svm_model.best_params_['kernel'])
            best_params['C'].append(svm_model.best_params_['C'])
            prediction = svm_model.best_estimator_.predict(x_test)
            accuracies.append(balanced_accuracy_score(prediction, y_test))

        return accuracies, best_params

    def kill_semantic_attributes(self, data, rate, new_value=None, limits=None):
        """
        Randomly sets to new_value a specific rate of the semantic attributes

        @param data: 2D numpy array with semantic data
        @param rate: float number from 0 to 1 specifying the rate of values to be replaced
        @return: 2D numpy array with new data set
        """
        num_sem_attrs = data.shape[1]

        new_data = np.copy(data)
        for ex in range(new_data.shape[0]):
            mask = [False] * data.shape[1]
            for idx in random.sample(range(data.shape[1]), round(num_sem_attrs * rate)):
                mask[idx] = True

            if limits is None and new_value is not None:
                new_data[ex, mask] = new_data[ex, mask] * 0 + new_value
            else:
                for i in range(len(mask)):
                    if mask[i]:
                        new_data[ex, i] = random.uniform(limits[0], limits[1])

        return new_data


if __name__ == '__main__':
    tag = 'awa'
    svm = SVMClassification(tag)
    s_dt, v_dt, lbs = svm.structure_data('../../../../Datasets/SAE/%s_demo_data.mat' % tag)

    for key_value in ['random', 0, 0.001, 0.1, -1, 9999]:
        n_folds = 10
        rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        acc = {key: {'acc': np.zeros(n_folds), 'mean': 0, 'std': 0, 'max': 0, 'min': 0} for key in rates}
        acc['ref'] = {'acc': None, 'params': None}
        acc['ref']['acc'], acc['ref']['params'] = svm.classify_data(s_dt, v_dt, lbs, n_folds)

        if key_value == 'random':
            key_value = None
            limits = (np.min(s_dt), np.max(s_dt))
        else:
            limits = None

        for rate in rates:
            s_dt = svm.kill_semantic_attributes(s_dt, rate=rate, new_value=key_value, limits=limits)
            acc[rate]['acc'], acc[rate]['params'] = svm.classify_data(s_dt, v_dt, lbs, n_folds)

            acc[rate]['mean'] = np.mean(acc[rate]['acc'])
            acc[rate]['std'] = np.std(acc[rate]['acc'])
            acc[rate]['max'] = np.max(acc[rate]['acc'])
            acc[rate]['min'] = np.min(acc[rate]['acc'])
            acc[rate]['acc'] = ', '.join(list(map(str, acc[rate]['acc'])))
            acc[rate]['params']['kernel'] = ', '.join(list(map(str, acc[rate]['params']['kernel'])))
            acc[rate]['params']['C'] = ', '.join(list(map(str, acc[rate]['params']['C'])))

        json_string = json.dumps(acc)
        with open('%s_svm_classification_%s.json' % tag, 'w+') as f:
            json.dump(json_string, f)

    # svm = SVMClassification('cub')
    # s_dt, v_dt, lbs = svm.structure_data('../../../../Datasets/SAE/cub_demo_data.mat')
    # acc, params = svm.classify_data(s_dt, v_dt, lbs, 10)
    #
    # print(params)
    # print(acc)
