"""
Computes the accuracy of zero shot learning classification and SVM classification
for AWA and CUB data sets. Semantic data is degraded to analyse its importance for
the classifications.
@author: Damares Resende
@contact: damaresresende@usp.br
@since: May 23, 2020
@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""
import json
import random
import numpy as np
from sklearn.svm import SVC
from scipy.io import loadmat
from sklearn.preprocessing import normalize
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from encoders.sec.src.autoencoder import Autoencoder, ModelType

from encoders.tools.src.utils import ZSL


class SemanticDegradation:
    def __init__(self, datafile, data_type, new_value=None, rates=None, ae_type='sae', epochs=50):
        """
        Initializes control variables

        :param datafile: string with path of data to load
        :param data_type: string to specify type of data: awa or cub
        :param new_value: real value to replace to. If not specified, a random value will be chosen
        :param rates: list of rates to test. Values must range from 0 to 1
        :param ae_type: type of autoencoder: sae or se
        :param epochs: number of epochs to use in training of AE of type se
        """
        self.data_type = data_type
        self.new_value = new_value
        self.data = loadmat(datafile)
        self.ae_type = ae_type
        self.epochs = epochs

        if rates is None:
            self.rates = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        else:
            self.rates = rates

        if new_value is None:
            self.limits = (np.min(self.data['S_tr']), np.max(self.data['S_tr']))
        else:
            self.limits = None

    def estimate_semantic_data(self, vis_tr_data, sem_tr_data, vis_te_data):
        """
        Trains SAE and applies it to visual data in order to estimate its correspondent semantic data

        :param vis_tr_data: visual training data
        :param sem_tr_data: semantic training data
        :param vis_te_data: visual test data
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

    def kill_semantic_attributes(self, data, rate):
        """
        Randomly sets to new_value a specific rate of the semantic attributes

        :param data: 2D numpy array with semantic data
        :param rate: float number from 0 to 1 specifying the rate of values to be replaced
        :return: 2D numpy array with new data set
        """
        num_sem_attrs = data.shape[1]

        new_data = np.copy(data)
        for ex in range(new_data.shape[0]):
            mask = [False] * data.shape[1]
            for idx in random.sample(range(data.shape[1]), round(num_sem_attrs * rate)):
                mask[idx] = True

            if self.limits is None and self.new_value is not None:
                new_data[ex, mask] = new_data[ex, mask] * 0 + self.new_value
            else:
                for i in range(len(mask)):
                    if mask[i]:
                        new_data[ex, i] = random.uniform(self.limits[0], self.limits[1])

        return new_data

    def structure_data_zsl(self):
        """
        Sets data of template labels, test labels, template semantic data and z_score flag
        according to the specified type of data to calculate SAE according to its original
        algorithm.

        :return: tuple with emp_labels, test_labels, s_te_pro and z_score
        """
        if self.data_type == 'awa':
            temp_labels = np.array([int(x) for x in self.data['param']['testclasses_id'][0][0]])
            test_labels = np.array([int(x) for x in self.data['param']['test_labels'][0][0]])
            s_te_pro = normalize(self.data['S_te_pro'].transpose(), norm='l2', axis=1, copy=True).transpose()

            return temp_labels, test_labels, s_te_pro, False

        elif self.data_type == 'cub':
            temp_labels = np.array([int(x) for x in self.data['te_cl_id']])
            test_labels = np.array([int(x) for x in self.data['test_labels_cub']])
            s_te_pro = self.data['S_te_pro']

            labels = list(map(int, self.data['train_labels_cub']))
            self.data['X_tr'], self.data['X_te'] = ZSL.dimension_reduction(self.data['X_tr'], self.data['X_te'], labels)

            return temp_labels, test_labels, s_te_pro, True

    def degrade_semantic_data_zsl(self, n_folds):
        """
        Randomly replaces the values of the semantic array for a new value specified and runs SAE over it.
        Saves the resultant accuracy in a dictionary. Data is degraded with rates ranging from 10 to 100%
        for a specific number of folds.

        :param n_folds: number of folds to use in cross validation
        :return: dictionary with classification accuracy for each fold
        """
        acc_dict = {key: {'acc': np.zeros(n_folds), 'mean': 0, 'std': 0, 'max': 0, 'min': 0} for key in self.rates}
        temp_labels, test_labels, s_te_pro, z_score = self.structure_data_zsl()

        for rate in self.rates:
            for j in range(n_folds):
                s_tr = self.kill_semantic_attributes(self.data['S_tr'], rate)

                if self.ae_type == 'sae':
                    s_te = self.estimate_semantic_data(self.data['X_tr'], s_tr, self.data['X_te'])
                elif self.ae_type == 'se':
                    if self.data_type == 'awa':
                        labels = self.data['param']['testclasses_id'][0][0]
                        train_labels = self.data['param']['train_labels'][0][0]

                        labels_dict = {labels[i][0]: attributes for i, attributes in enumerate(self.data['S_te_pro'])}
                        sem_data = np.array([labels_dict[label[0]] for label in self.data['param']['test_labels'][0][0]])
                    elif self.data_type == 'cub':
                        labels = self.data['te_cl_id']
                        train_labels = self.data['train_labels_cub']

                        labels_dict = {labels[i][0]: attributes for i, attributes in enumerate(self.data['S_te_pro'])}
                        sem_data = np.array([labels_dict[label[0]] for label in self.data['test_labels_cub']])
                    else:
                        raise ValueError('Invalid type of data')

                    input_length = output_length = self.data['X_tr'].shape[1] + self.data['S_tr'].shape[1]
                    ae = Autoencoder(input_length, self.data['S_tr'].shape[1], output_length, ModelType.SIMPLE_AE, self.epochs)
                    s_te = ae.estimate_semantic_data(self.data['X_tr'], self.data['S_tr'], self.data['X_te'], sem_data, train_labels)
                else:
                    raise ValueError('Invalid type of autoencoder')

                acc, _ = ZSL.zsl_el(s_te, s_te_pro, test_labels, temp_labels, 1, z_score)
                acc_dict[rate]['acc'][j] = acc

            acc_dict[rate]['mean'] = np.mean(acc_dict[rate]['acc'])
            acc_dict[rate]['std'] = np.std(acc_dict[rate]['acc'])
            acc_dict[rate]['max'] = np.max(acc_dict[rate]['acc'])
            acc_dict[rate]['min'] = np.min(acc_dict[rate]['acc'])
            acc_dict[rate]['acc'] = ', '.join(list(map(str, acc_dict[rate]['acc'])))

        return acc_dict

    def structure_data_svm(self):
        """
        Loads data and structures it in a unique set of semantic data, visual data and labels,
        so SVM can be applied.

        :return: tuple with arrays for semantic data, visual data and labels
        """
        if self.data_type == 'awa':
            y_tr = self.data['param']['train_labels'][0][0]
            y_te = self.data['param']['test_labels'][0][0]
            attr_id = self.data['param']['testclasses_id'][0][0]
            x_tr, x_te = self.data['X_tr'], self.data['X_te']
        elif self.data_type == 'cub':
            y_tr = self.data['train_labels_cub']
            y_te = self.data['test_labels_cub']
            attr_id = self.data['te_cl_id']
            x_tr, x_te = ZSL.dimension_reduction(self.data['X_tr'], self.data['X_te'], list(map(int, y_tr)))
        else:
            raise ValueError('Unknown type of data')

        labels_dict = {attr_id[i][0]: attributes for i, attributes in enumerate(self.data['S_te_pro'])}
        s_te = np.array([labels_dict[label[0]] for label in y_te])

        return np.vstack((self.data['S_tr'], s_te)), np.vstack((x_tr, x_te)), np.vstack((y_tr, y_te))[:, 0]

    def degrade_semantic_data_svm(self, n_folds):
        """
        Trains SVM classifier using grid search and k-fold cross validation. Test data is
        randomly replaced by a random value. The amount of data replaced varies from 0 to 100%
        according to the specified rate.

        :param n_folds: number of folds to use in cross validation
        :return: dictionary with classification accuracy for each fold
        """
        acc_dict = {key: {'acc': [], 'mean': 0, 'std': 0, 'max': 0, 'min': 0, 'C': []} for key in self.rates}
        sem_data, vis_data, labels = self.structure_data_svm()
        tuning_params = {'kernel': ['linear'], 'C': [0.5, 1, 5, 10]}

        for rate in self.rates:
            skf = StratifiedKFold(n_splits=n_folds, random_state=None, shuffle=True)
            for train_index, test_index in skf.split(sem_data, labels):
                x_train = sem_data[train_index]

                if self.ae_type == 'sae':
                    x_test = self.estimate_semantic_data(vis_data[train_index], sem_data[train_index], vis_data[test_index])
                elif self.ae_type == 'se':
                    input_length = output_length = vis_data.shape[1] + sem_data.shape[1]
                    ae = Autoencoder(input_length, sem_data.shape[1], output_length, ModelType.SIMPLE_AE, self.epochs)
                    x_test = ae.estimate_semantic_data(vis_data[train_index], sem_data[train_index], vis_data[test_index], sem_data[test_index], labels[train_index])
                else:
                    raise ValueError('Invalid type of autoencoder')

                y_train, y_test = labels[train_index], labels[test_index]
                svm_model = GridSearchCV(SVC(gamma='scale'), tuning_params, scoring='recall_macro', n_jobs=-1)
                svm_model.fit(x_train, y_train)

                svm_acc = []
                for _ in range(n_folds):
                    prediction = svm_model.best_estimator_.predict(self.kill_semantic_attributes(x_test, rate))
                    svm_acc.append(balanced_accuracy_score(prediction, y_test))

                acc_dict[rate]['acc'].append(np.mean(np.array(svm_acc)))
                acc_dict[rate]['C'].append(svm_model.best_params_['C'])

            acc_dict[rate]['acc'] = np.array(acc_dict[rate]['acc'])
            acc_dict[rate]['mean'] = np.mean(acc_dict[rate]['acc'])
            acc_dict[rate]['std'] = np.std(acc_dict[rate]['acc'])
            acc_dict[rate]['max'] = np.max(acc_dict[rate]['acc'])
            acc_dict[rate]['min'] = np.min(acc_dict[rate]['acc'])
            acc_dict[rate]['acc'] = ', '.join(list(map(str, acc_dict[rate]['acc'])))
            acc_dict[rate]['C'] = ', '.join(list(map(str, acc_dict[rate]['C'])))

        return acc_dict

    @staticmethod
    def write2json(acc_dict, filename):
        """
        Writes data from accuracy dictionary to JSON file

        :param acc_dict: dict with classification accuracies
        :param filename: string with name of file to write data to
        :return: None
        """
        json_string = json.dumps(acc_dict)
        with open(filename, 'w+') as f:
            json.dump(json_string, f)
