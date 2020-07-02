import os
import json
import random
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import normalize

from encoders.tools.src.utils import ZSL
from encoders.sec.src.autoencoder import Autoencoder, ModelType


class AwAZSL:
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
        Sets data of template labels, test labels, template semantic data and z_score flag
        according to the specified type of data to calculate SAE according to its original
        algorithm.

        :param data: data dictionary
        :return: tuple with emp_labels, test_labels, s_te_pro and z_score
        """
        temp_labels = np.array([int(x) for x in data['param']['testclasses_id'][0][0]])
        test_labels = np.array([int(x) for x in data['param']['test_labels'][0][0]])
        train_labels = np.array([int(x) for x in data['param']['train_labels'][0][0]])
        s_te_pro = normalize(data['S_te_pro'].transpose(), norm='l2', axis=1, copy=True).transpose()

        labels_dict = {temp_labels[i]: attributes for i, attributes in enumerate(s_te_pro)}
        sem_te_data = np.array([labels_dict[label] for label in test_labels])

        return temp_labels, train_labels, test_labels, s_te_pro, sem_te_data, False


class CUBZSL:
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

        labels = list(map(int, data['train_labels_cub']))
        data['X_tr'], data['X_te'] = ZSL.dimension_reduction(data['X_tr'], data['X_te'], labels)

        labels_dict = {temp_labels[i]: attributes for i, attributes in enumerate(s_te_pro)}
        sem_te_data = np.array([labels_dict[label] for label in test_labels])

        return temp_labels, train_labels, test_labels, s_te_pro, sem_te_data, True


class SemanticDegradation:
    def __init__(self, datafile, data_type, new_value=None, rates=None, results_path='.'):
        """
        Initializes control variables

        :param datafile: string with path of data to load
        :param data_type: string with data type: awa or cub
        :param new_value: real value to replace to. If not specified, a random value will be chosen
        :param rates: list of rates to test. Values must range from 0 to 1
        :param results_path: string that indicates where results will be saved
        """
        self.new_value = new_value
        self.data = loadmat(datafile)
        self.results_path = results_path
        self.data_type = data_type

        if data_type == 'awa':
            self.dealer = AwAZSL()
        elif data_type == 'cub':
            self.dealer = CUBZSL()
        else:
            raise ValueError('Invalid data type. It should be awa or cub')

        self.rates = rates if rates else [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        self.results = {rate: dict() for rate in self.rates}

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

    def degrade_semantic_data(self, n_folds, ae_type, epochs=50):
        """
        Randomly replaces the values of the semantic array for a new value specified and runs SAE or SEC over it.
        Saves the resultant accuracy in a dictionary. Data is degraded with rates ranging from 10 to 100%
        for a specific number of folds.

        :param n_folds: number of folds to use in cross validation
        :param ae_type: string with ae type: sec or sae
        :param epochs: number of epochs to use in training of AE of type se
        :return: dictionary with classification accuracy for each fold
        """
        acc_dict = {key: {'acc': np.zeros(n_folds), 'mean': 0, 'std': 0, 'max': 0, 'min': 0} for key in self.rates}
        temp_labels, tr_labels, test_labels, s_te_pro, sem_te_data, z_score = self.dealer.structure_data(self.data)

        for rate in self.rates:
            self.results = dict()
            str_rate = str(round(rate * 100))
            for j in range(n_folds):
                if ae_type == 'sae':
                    s_tr = self.kill_semantic_attributes(self.data['S_tr'], rate)
                    s_te = self.dealer.estimate_semantic_data_sae(self.data['X_tr'], s_tr, self.data['X_te'])
                elif ae_type == 'sec':
                    s_te = self.kill_semantic_attributes(sem_te_data, rate)
                    w_info = {'label': 'fold_%d' % (j + 1), 'path': os.path.join(self.results_path, str_rate)}
                    s_te, summary = self.dealer.estimate_semantic_data_sec(self.data['X_tr'], self.data['S_tr'],
                                                                           self.data['X_te'], s_te,
                                                                           tr_labels, epochs, w_info)
                    self.results['fold_%d' % (j + 1)] = summary
                else:
                    raise ValueError('Invalid type of autoencoder. Accepted values are sec or sae')

                acc, _ = ZSL.zsl_el(s_te, s_te_pro, test_labels, temp_labels, 1, z_score)
                acc_dict[rate]['acc'][j] = acc

            acc_dict[rate]['mean'] = self.results['mean'] = np.mean(acc_dict[rate]['acc'])
            acc_dict[rate]['std'] = self.results['std'] = np.std(acc_dict[rate]['acc'])
            acc_dict[rate]['max'] = self.results['max'] = np.max(acc_dict[rate]['acc'])
            acc_dict[rate]['min'] = self.results['min'] = np.min(acc_dict[rate]['acc'])
            acc_dict[rate]['acc'] = self.results['acc'] = ', '.join(list(map(str, acc_dict[rate]['acc'])))

            if not os.path.isdir(os.path.join(self.results_path, str_rate)):
                os.mkdir(os.path.join(self.results_path, str_rate))
            name = '%s_summary_v2s_%s_%s.json' % (self.data_type, str_rate, ae_type)
            self.write2json(self.results, os.sep.join([self.results_path, str_rate, name]))

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
