"""
Loads data saved from the extraction and saves it in a Matlab structure
so SAE code can be evaluated on it

@author: Damares Resende
@contact: damaresresende@usp.br
@since: June 15, 2020

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""
import numpy as np
from os import path
from scipy.io import savemat


class Parser:
    def __init__(self, data_type):
        """
        Sets test labels and data type

        :param data_type: type of data analyzed: awa or cub.
        """
        self.data_type = data_type
        self.data_dict = dict()

        if self.data_type == 'AWA2':
            self.test_data = False
            self.test_labels = [25, 39, 15,  6, 42, 14, 18, 48, 34, 24]
        elif self.data_type == 'CUB200':
            self.test_data = True
            self.test_labels = [1, 4, 6, 8, 9, 14, 23, 29, 31, 33, 34, 35, 36, 37, 38, 43, 49, 51, 53, 66, 72, 79,
                                83, 84, 86, 91, 95, 96, 98, 101, 102, 103, 112, 114, 119, 121, 130, 135, 138, 147,
                                156, 163, 165, 166, 180, 183, 185, 186, 187, 197]
        else:
            raise ValueError('Unknown type of data')

    def _build_masks(self, labels):
        """
        Builds a boolean mask based on the test labels set and the data set labels

        :param labels: data set labels
        :return: tuple with training mask array and test mask array
        """
        tr_mask = [False if lb in self.test_labels else True for lb in labels]
        te_mask = [True if lb in self.test_labels else False for lb in labels]

        return tr_mask, te_mask

    def split_data(self, vis_data, sem_data, labels):
        """
        Splits data into training and testing data and saves results in the data dictionary

        :param vis_data: 2D numpy array with visual data
        :param sem_data: 2D numpy array with semantic data
        :param labels: array with labels
        :return: None
        """
        tr_mask, te_mask = self._build_masks(labels)
        self.data_dict['S_tr'] = sem_data[tr_mask]
        self.data_dict['S_te'] = sem_data[te_mask]

        self.data_dict['X_tr'] = vis_data[tr_mask]
        self.data_dict['X_te'] = vis_data[te_mask]
        self.data_dict['S_te_pro'] = self.build_semantic_matrix(sem_data, labels)

        if self.data_type == 'AWA2':
            self.data_dict['S_te_gt'] = self.build_semantic_matrix(sem_data, labels)
            self.data_dict['param'] = dict()
            self.data_dict['param']['testclasses_id'] = np.expand_dims(np.array(self.test_labels), axis=1)
            self.data_dict['param']['test_labels'] = np.expand_dims(np.array(labels[te_mask]), axis=1)
            self.data_dict['param']['train_labels'] = np.expand_dims(np.array(labels[tr_mask]), axis=1)

        elif self.data_type == 'CUB200':
            self.data_dict['te_cl_id'] = np.expand_dims(np.array(self.test_labels), axis=1)
            self.data_dict['test_labels_cub'] = np.expand_dims(np.array(labels[te_mask]), axis=1)
            self.data_dict['train_labels_cub'] = np.expand_dims(np.array(labels[tr_mask]), axis=1)

    def build_semantic_matrix(self, sem_data, labels):
        """
        Builds semantic data matrix based on the test set labels and semantic attributes. Each row of
        the matrix is an array of attributes for a specific label in the test labels set.

        :param sem_data: 2D numpy array with semantic data
        :param labels: list of labels for the semantic data specified
        :return: 2D numpy array with semantic matrix
        """
        s_te_pro = []
        for lb in self.test_labels:
            for i, att in enumerate(sem_data):
                if labels[i] == lb:
                    s_te_pro.append(att)
                    break

        return np.array(s_te_pro)

    def save_data(self, file_name):
        """
        Saves data dictionary into a .mat file

        :param file_name: file name and path
        :return: None
        """
        savemat(file_name, self.data_dict)

    def load_data(self, base_path):
        """
        Loads data into 2D numpy arrays for visual data, semantic data and labels

        :param base_path: string with path where data files were saved
        :return: tuple with visual data, semantic data and labels
        """
        with open(path.join(base_path, '%s_x_train_sem.txt' % self.data_type)) as f:
            sem_data = [list(map(float, line.split(' '))) for line in f.readlines()]

        if self.test_data:
            with open(path.join(base_path, '%s_x_test_sem.txt' % self.data_type)) as f:
                sem_data.extend([list(map(float, line.split(' '))) for line in f.readlines()])

        with open(path.join(base_path, '%s_x_train_vis.txt' % self.data_type)) as f:
            vis_data = [list(map(float, line.split(' '))) for line in f.readlines()]

        if self.test_data:
            with open(path.join(base_path, '%s_x_test_vis.txt' % self.data_type)) as f:
                vis_data.extend([list(map(float, line.split(' '))) for line in f.readlines()])

        with open(path.join(base_path, '%s_y_train.txt' % self.data_type)) as f:
            labels = [int(line.strip()) for line in f.readlines()]

        if self.test_data:
            with open(path.join(base_path, '%s_y_test.txt' % self.data_type)) as f:
                labels.extend([int(line.strip()) for line in f.readlines()])

        return np.array(vis_data), np.array(sem_data), np.array(labels)
