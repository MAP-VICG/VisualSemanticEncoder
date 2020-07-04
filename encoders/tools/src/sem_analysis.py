"""
Applies multiclass classification or zero shot learning classification to a
specified data. Classification input is the semantic data array. Semantic data
is computed either using SAE algorithm, proposed by Elyor Kodirov, Tao Xiang,
and Shaogang Gong, or SEC algorithm, proposed by Damares Resende and Moacir Ponti

@author: Damares Resende
@contact: damaresresende@usp.br
@since: July 2, 2020

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""
import os
import json
import random
import numpy as np
from scipy.io import loadmat
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

from encoders.tools.src.utils import ZSL
from encoders.sec.src.autoencoder import Autoencoder, ModelType


class AwAClassification:
    @staticmethod
    def estimate_semantic_data_sae(vis_tr_data, sem_tr_data, vis_te_data):
        """
        Trains SAE and applies it to visual data in order to estimate its correspondent semantic data

        :param vis_tr_data: visual training data
        :param sem_tr_data: semantic training data
        :param vis_te_data: visual test data
        :return: 2D numpy array with estimated semantic data
        """
        x_tr = normalize(vis_tr_data.transpose(), norm='l2', axis=1, copy=True).transpose()
        w = ZSL.sae(x_tr.transpose(), sem_tr_data.transpose(), 500000)
        return vis_te_data.dot(normalize(w, norm='l2', axis=1, copy=True).transpose())

    @staticmethod
    def estimate_semantic_data_sec(vis_tr_data, sem_tr_data, vis_te_data, sem_te_data, tr_labels, epochs, w_info):
        """
        Trains AE and applies it to visual data in order to estimate its correspondent semantic data

        :param vis_tr_data: visual training data
        :param sem_tr_data: semantic training data
        :param vis_te_data: visual test data
        :param sem_te_data: visual test data
        :param tr_labels: training labels
        :param epochs: number of epochs
        :param w_info: dictionary with path to save weights file and the weights label
        :return: 2D numpy array with estimated semantic data
        """
        input_length = output_length = vis_tr_data.shape[1] + sem_tr_data.shape[1]
        ae = Autoencoder(input_length, sem_tr_data.shape[1], output_length, ModelType.SIMPLE_AE, epochs)
        _, s_te = ae.estimate_semantic_data(vis_tr_data, sem_tr_data, vis_te_data, sem_te_data, tr_labels)
        ae.save_best_weights('awa_v2s_%s' % w_info['label'], w_info['path'])

        return s_te, ae.get_summary()

    @staticmethod
    def structure_data(data):
        """
        Sets data of template labels, test labels, template semantic data and z_score flag according to
        the specified type of data to calculate SAE according to its original algorithm.

        :param data: data dictionary
        :return: tuple with temp_labels, train_labels, test_labels, s_te_pro, sem_te_data and z_score
        """
        temp_labels = np.array([int(x) for x in data['param']['testclasses_id'][0][0]])
        test_labels = np.array([int(x) for x in data['param']['test_labels'][0][0]])
        train_labels = np.array([int(x) for x in data['param']['train_labels'][0][0]])
        s_te_pro = normalize(data['S_te_pro'].transpose(), norm='l2', axis=1, copy=True).transpose()

        labels_dict = {temp_labels[i]: attributes for i, attributes in enumerate(s_te_pro)}
        sem_te_data = np.array([labels_dict[label] for label in test_labels])

        return temp_labels, train_labels, test_labels, s_te_pro, sem_te_data, False

    @staticmethod
    def get_classification_data(data):
        """
        Joins training and test sets of the original data structure, so it can be used in k fold
        cross validation. For simple classification applications it is necessary because training
        and test sets cannot be disjoint

        :param data: data dictionary
        :return: tuple with sem_data, vis_data and data_labels
        """
        _, train_labels, test_labels, _, sem_te_data, _ = AwAClassification.structure_data(data)

        sem_data = np.vstack((data['S_tr'], sem_te_data))
        vis_data = np.vstack((data['X_tr'], data['X_te']))
        data_labels = np.vstack((np.expand_dims(train_labels, axis=1), np.expand_dims(test_labels, axis=1)))

        return sem_data, vis_data, data_labels.reshape(-1)


class CUBClassification:
    @staticmethod
    def estimate_semantic_data_sae(vis_tr_data, sem_tr_data, vis_te_data):
        """
        Trains SAE and applies it to visual data in order to estimate its correspondent semantic data

        :param vis_tr_data: visual training data
        :param sem_tr_data: semantic training data
        :param vis_te_data: visual test data
        :return: 2D numpy array with estimated semantic data
        """
        s_tr = normalize(sem_tr_data, norm='l2', axis=1, copy=False)
        w = ZSL.sae(vis_tr_data.transpose(), s_tr.transpose(), .2).transpose()
        return vis_te_data.dot(w)

    @staticmethod
    def estimate_semantic_data_sec(vis_tr_data, sem_tr_data, vis_te_data, sem_te_data, tr_labels, epochs, w_info):
        """
        Trains AE and applies it to visual data in order to estimate its correspondent semantic data

        :param vis_tr_data: visual training data
        :param sem_tr_data: semantic training data
        :param vis_te_data: visual test data
        :param sem_te_data: visual test data
        :param tr_labels: training labels
        :param epochs: number of epochs
        :param w_info: dictionary with path to save weights file and the weights label
        :return: 2D numpy array with estimated semantic data
        """
        input_length = output_length = vis_tr_data.shape[1] + sem_tr_data.shape[1]
        ae = Autoencoder(input_length, sem_tr_data.shape[1], output_length, ModelType.SIMPLE_AE, epochs)
        _, s_te = ae.estimate_semantic_data(vis_tr_data, sem_tr_data, vis_te_data, sem_te_data, tr_labels)
        ae.save_best_weights('cub_v2s_%s' % w_info['label'], w_info['path'])

        return s_te, ae.get_summary()

    @staticmethod
    def structure_data(data):
        """
        Sets data of template labels, test labels, template semantic data and z_score flag
        according to the specified type of data to calculate SAE according to its original
        algorithm.

        :param data: data dictionary
        :return: tuple with emp_labels, test_labels, s_te_pro and z_score
        """
        temp_labels = np.array([int(x) for x in data['te_cl_id']])
        test_labels = np.array([int(x) for x in data['test_labels_cub']])
        train_labels = np.array([int(x) for x in data['train_labels_cub']])
        s_te_pro = data['S_te_pro']

        labels_dict = {temp_labels[i]: attributes for i, attributes in enumerate(s_te_pro)}
        sem_te_data = np.array([labels_dict[label] for label in test_labels])

        return temp_labels, train_labels, test_labels, s_te_pro, sem_te_data, True

    @staticmethod
    def get_classification_data(data):
        """
        Joins training and test sets of the original data structure, so it can be used in k fold
        cross validation. For simple classification applications it is necessary because training
        and test sets cannot be disjoint

        :param data: data dictionary
        :return: tuple with sem_data, vis_data and data_labels
        """
        _, train_labels, test_labels, _, sem_te_data, _ = CUBClassification.structure_data(data)

        sem_data = np.vstack((data['S_tr'], sem_te_data))
        vis_data = np.vstack((data['X_tr'], data['X_te']))
        data_labels = np.vstack((np.expand_dims(train_labels, axis=1), np.expand_dims(test_labels, axis=1)))

        return sem_data, vis_data, data_labels.reshape(-1)


class SemanticDegradation:
    def __init__(self, datafile, new_value=None, rates=None, results_path='.'):
        """
        Initializes control variables

        :param datafile: string with path of data to load
        :param new_value: real value to replace to. If not specified, a random value will be chosen
        :param rates: list of rates to test. Values must range from 0 to 1
        :param results_path: string that indicates where results will be saved
        """
        self.new_value = new_value
        self.data = loadmat(datafile)
        self.results_path = results_path

        self.epochs = self.n_folds = 0
        self.ae_type = self.dealer = self.data_type = None

        self.rates = rates if rates else [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        self.results = {rate: dict() for rate in self.rates}
        self.acc_dict = dict()

    def kill_semantic_attributes(self, data, rate):
        """
        Randomly sets to new_value a specific rate of the semantic attributes

        :param data: 2D numpy array with semantic data
        :param rate: float number from 0 to 1 specifying the rate of values to be replaced
        :return: 2D numpy array with new data set
        """
        new_data = np.copy(data)
        if self.new_value:
            for ex in range(new_data.shape[0]):
                for idx in random.sample(range(data.shape[1]), round(data.shape[1] * rate)):
                    new_data[ex, idx] = self.new_value
        else:
            for ex in range(new_data.shape[0]):
                for idx in random.sample(range(data.shape[1]), round(data.shape[1] * rate)):
                    new_data[ex, idx] = random.uniform(np.max(self.data['S_tr']), np.max(self.data['S_tr']))

        return new_data

    def degrade_semantic_data(self, data_type, class_type, n_folds, epochs=50):
        """
        Randomly replaces the values of the semantic array for a new value specified and runs SAE or SEC over it.
        Saves the resultant accuracy in a dictionary. Data is degraded with rates ranging from 10 to 100%
        for a specific number of folds.

        :param data_type: string with data type: awa or cub
        :param class_type: string with classification type: zsl or cls
        :param n_folds: number of folds to use in cross validation
        :param epochs: number of epochs to use in training of AE of type se
        :return: dictionary with classification accuracy for each fold
        """
        self._set_training_params(data_type, n_folds, epochs)
        acc_dict = {key: {'acc': np.zeros(n_folds), 'mean': 0, 'std': 0, 'max': 0, 'min': 0} for key in self.rates}

        if class_type == 'zsl':
            temp_labels, tr_labels, te_labels, s_te_pro, s_te_data, z_score = self.dealer.structure_data(self.data)
            if self.data_type == 'cub':
                labels = list(map(int, self.data['train_labels_cub']))
                x_tr, x_te = ZSL.dimension_reduction(self.data['X_tr'], self.data['X_te'], labels)
            else:
                x_tr, x_te = self.data['X_tr'], self.data['X_te']

            s_te = self.dealer.estimate_semantic_data_sae(x_tr, self.data['S_tr'], x_te)
            acc_dict['ref'], _ = ZSL.zsl_el(s_te, s_te_pro, te_labels, temp_labels, 1, z_score)
        elif class_type == 'cls':
            accuracies = []
            skf = StratifiedKFold(n_splits=self.n_folds, random_state=None, shuffle=True)
            sem_data, vis_data, data_labels = self.dealer.get_classification_data(self.data)

            for tr_index, te_index in skf.split(sem_data, data_labels):
                if self.data_type == 'cub':
                    x_tr, x_te = ZSL.dimension_reduction(vis_data[tr_index], vis_data[te_index], data_labels[tr_index])
                else:
                    x_tr, x_te = vis_data[tr_index], vis_data[te_index]

                s_te = self.dealer.estimate_semantic_data_sae(x_tr, sem_data[tr_index], x_te)
                clf = make_pipeline(StandardScaler(), SVC(gamma='auto', C=1.0, kernel='linear'))
                clf.fit(sem_data[tr_index], data_labels[tr_index])

                prediction = clf.predict(s_te)
                accuracies.append(balanced_accuracy_score(prediction, data_labels[te_index]))

            acc_dict['ref'] = np.mean(np.array(accuracies))

        for rate in self.rates:
            self.results = dict()
            str_rate = str(round(rate * 100)) if rate > 10 else '0' + str(round(rate * 100))

            if class_type == 'zsl':
                temp_labels, tr_labels, te_labels, s_te_pro, s_te_data, z_score = self.dealer.structure_data(self.data)
                acc_dict[rate]['acc'] = self._zero_shot_learning(temp_labels, tr_labels, te_labels, s_te_pro, s_te_data, z_score, rate)
            elif class_type == 'cls':
                sem_data, vis_data, data_labels = self.dealer.get_classification_data(self.data)
                acc_dict[rate]['acc'] = self._simple_classification(sem_data, vis_data, data_labels, rate)
            else:
                raise ValueError('Unknown type of classification analysis')

            acc_dict[rate]['mean'] = self.results['mean'] = np.mean(acc_dict[rate]['acc'])
            acc_dict[rate]['std'] = self.results['std'] = np.std(acc_dict[rate]['acc'])
            acc_dict[rate]['max'] = self.results['max'] = np.max(acc_dict[rate]['acc'])
            acc_dict[rate]['min'] = self.results['min'] = np.min(acc_dict[rate]['acc'])
            acc_dict[rate]['acc'] = self.results['acc'] = ', '.join(list(map(str, acc_dict[rate]['acc'])))

            if not os.path.isdir(os.path.join(self.results_path, str_rate)):
                os.mkdir(os.path.join(self.results_path, str_rate))

            name = '%s_summary_v2s_%s.json' % (data_type, str_rate)
            self.write2json(self.results, os.sep.join([self.results_path, str_rate, name]))

        return acc_dict

    def _set_training_params(self, data_type, n_folds, epochs):
        """
        Sets configuration parameters to perform training

        :param data_type: string with data type: awa or cub
        :param n_folds: number of folds
        :param epochs: number of epochs
        :return:
        """
        self.epochs = epochs
        self.n_folds = n_folds
        self.data_type = data_type

        if data_type == 'awa':
            self.dealer = AwAClassification()
        elif data_type == 'cub':
            self.dealer = CUBClassification()
        else:
            raise ValueError('Invalid data type. It should be awa or cub')

    def _zero_shot_learning(self, temp_labels, tr_labels, te_labels, s_te_pro, s_te_data, z_score, rate=0):
        """
        Applies zero shot learning classification n_folds times to the given data, and saves the results.

        :param temp_labels: set of label ids for test data
        :param tr_labels: training set labels
        :param te_labels: test set labels
        :param s_te_pro: matrix of semantic data per class for test set
        :param s_te_data: semantic data for test
        :param z_score: if true, computes zscore of distance vector
        :param rate: rate to degrade semantic data
        :return: list of accuracies for ZSL
        """
        accuracies = np.zeros(self.n_folds)
        str_rate = str(round(rate * 100)) if rate > 10 else '0' + str(round(rate * 100))

        if self.data_type == 'cub':
            labels = list(map(int, self.data['train_labels_cub']))
            x_tr, x_te = ZSL.dimension_reduction(self.data['X_tr'], self.data['X_te'], labels)
        else:
            x_tr, x_te = self.data['X_tr'], self.data['X_te']

        for j in range(self.n_folds):
            s_tr = self.data['S_tr']
            s_te = self.kill_semantic_attributes(s_te_data, rate)
            info = {'label': 'fold_%d' % (j + 1), 'path': os.path.join(self.results_path, str_rate)}
            s_te, summary = self.dealer.estimate_semantic_data_sec(x_tr, s_tr, x_te, s_te, tr_labels, self.epochs, info)

            self.results['fold_%d' % (j + 1)] = summary
            accuracies[j], _ = ZSL.zsl_el(s_te, s_te_pro, te_labels, temp_labels, 1, z_score)
        return accuracies

    def _simple_classification(self, sem_data, vis_data, data_labels, rate=0):
        """
        Applies SVM classification on the specified data using k fold cross validation and degrading
        semantic data at the specified rate.

        :param sem_data: semantic data
        :param vis_data:  visual data
        :param data_labels: data labels
        :param rate: degradation rate
        :return:
        """
        fold = 0
        accuracies = np.zeros(self.n_folds)
        skf = StratifiedKFold(n_splits=self.n_folds, random_state=None, shuffle=True)
        str_rate = str(round(rate * 100)) if rate > 10 else '0' + str(round(rate * 100))

        for tr_index, te_index in skf.split(sem_data, data_labels):
            if self.data_type == 'cub':
                x_tr, x_te = ZSL.dimension_reduction(vis_data[tr_index], vis_data[te_index], data_labels[tr_index])
            else:
                x_tr, x_te = vis_data[tr_index], vis_data[te_index]

            s_tr = sem_data[tr_index]
            lb_tr = data_labels[tr_index]
            s_te = self.kill_semantic_attributes(sem_data[te_index], rate)
            w_info = {'label': 'fold_%d' % (fold + 1), 'path': os.path.join(self.results_path, str_rate)}

            s_te, summary = self.dealer.estimate_semantic_data_sec(x_tr, s_tr, x_te, s_te, lb_tr, self.epochs, w_info)
            self.results['fold_%d' % (fold + 1)] = summary

            clf = make_pipeline(StandardScaler(), SVC(gamma='auto', C=1.0, kernel='linear'))
            clf.fit(sem_data[tr_index], data_labels[tr_index])

            prediction = clf.predict(s_te)
            accuracies[fold] = balanced_accuracy_score(prediction, data_labels[te_index])
            fold += 1
        return accuracies

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
