"""
Parsers the data made available by Elyor Kodirov, Tao Xiang, and Shaogang Gong in the
paper "Semantic Autoencoder for Zero-shot Learning" published in CVPR 2017, to a new data
structure used in this project. The reference code was originally written in Matlab and
is here being transformed to Python.

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Apr 24, 2021

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""
import os
import bz2
import pickle
import argparse
import numpy as np
from pathlib import Path
from scipy.io import loadmat


class Matlab2PickleParser:
    def __init__(self, data_path):
        """
        Loads data and defines new data structures

        :param data_path: string with path pointing to the .mat file with Kodirov's dataset
        """
        self.new_data_tr = dict()
        self.new_data_te = dict()
        self.data = loadmat(data_path)

        self.new_data_tr['X_tr'] = self.data['X_tr'].astype(np.float64)
        self.new_data_tr['S_tr'] = self.data['S_tr'].astype(np.float64)
        self.new_data_te['X_te'] = self.data['X_te'].astype(np.float64)
        self.new_data_te['S_te_pro'] = self.data['S_te_pro'].astype(np.float64)

    def __call__(self, file_path, file_name):
        """
        Parses the data in the .mat file into a new format and saves it in a compressed pickle file.
        The data is split into two files, one with training data, another with test data. This is done
        aiming to make files smaller.

        :param file_path: string with path where output will be saved
        :param file_name: string with the file name of the dataset
        """
        self.new_data_tr['Y_tr'] = self._get_tr_labels()
        self.new_data_te['Y_te'] = self._get_te_labels()
        self.new_data_te['S_te'] = self._get_te_sem_data()
        self.new_data_te['S_te_pro_lb'] = self._get_te_pro_labels()
        self._save_data(file_path, file_name)

    def _get_te_sem_data(self):
        """
        Parses the semantic data in test set. It changes depending on the dataset being parsed so this method
        must be overloaded.

        :return: None
        """
        return None

    def _get_te_labels(self):
        """
        Parses the labels in test set. It changes depending on the dataset being parsed so this method
        must be overloaded.

        :return: None
        """
        return None

    def _get_tr_labels(self):
        """
        Parses the labels in training set. It changes depending on the dataset being parsed so this method
        must be overloaded.

        :return: None
        """
        return None

    def _get_te_pro_labels(self):
        """
        Parses the labels in the prototype matrix. It changes depending on the dataset being parsed so this method
        must be overloaded. In this case it is important to keep the order of the labels for correct mapping later.

        :return: None
        """
        return None

    def _save_data(self, file_path, file_name):
        """
        Saves the data parsed and structure into new_data_tr and new_data_te dictionaries. These dictionaries
        are saved in different files. This is done in order to keep the files small enough to push them to GitHub.

        :return: None
        """
        Path(file_path).mkdir(parents=True, exist_ok=True)

        with bz2.BZ2File(os.sep.join([file_path, file_name]) + '_tr.pbz2', 'w') as f:
            pickle.dump(self.new_data_tr, f)

        with bz2.BZ2File(os.sep.join([file_path, file_name]) + '_te.pbz2', 'w') as f:
            pickle.dump(self.new_data_te, f)


class AWAParser(Matlab2PickleParser):
    def _get_te_labels(self):
        """
        Parses the test labels.

        :return: 1D numpy array with the test labels for each instance in the test set
        """
        return np.array([label[0] for label in self.data['param']['test_labels'][0][0]], dtype=np.uint8)

    def _get_tr_labels(self):
        """
        Parses the training labels.

        :return: 1D numpy array with the training labels for each instance in the training set
        """
        return np.array([label[0] for label in self.data['param']['train_labels'][0][0]], dtype=np.uint8)

    def _get_te_sem_data(self):
        """
        Parses the semantic data for the test set. This data is not directly structured in the .mat file
        Kodirov has provided, and it is not necessary for zero-shot learning, but it is needed for classification.
        Here we build the S_te dataset based on the test labels, the prototype labels and the semantic prototype
        matrix for the test set.

        :return: 2D numpy array with semantic prototype for each instance in the test set
        """
        lbs = {self.data['param']['testclasses_id'][0][0][j][0]: attrs for j, attrs in enumerate(self.data['S_te_pro'])}
        return np.array([lbs[label[0]] for label in self.data['param']['test_labels'][0][0]], dtype=np.float64)

    def _get_te_pro_labels(self):
        """
        Parses the labels in the prototype matrix. In this case it is important to keep the order of the labels
        for correct mapping later.

        :return: 1D numpy array with the labels for each semantic prototype in S_te_pro matrix
        """
        return np.array([int(x) for x in self.data['param']['testclasses_id'][0][0]])


class CUBParser(Matlab2PickleParser):
    def _get_te_labels(self):
        """
        Parses the test labels.

        :return: 1D numpy array with the test labels for each instance in the test set
        """
        return np.array([lb[0] for lb in self.data['test_labels_cub']], dtype=np.uint8)

    def _get_tr_labels(self):
        """
        Parses the training labels.

        :return: 1D numpy array with the training labels for each instance in the training set
        """
        return np.array([lb[0] for lb in self.data['train_labels_cub']], dtype=np.uint8)

    def _get_te_sem_data(self):
        """
        Parses the semantic data for the test set. This data is not directly structured in the .mat file
        Kodirov has provided, and it is not necessary for zero-shot learning, but it is needed for classification.
        Here we build the S_te dataset based on the test labels, the prototype labels and the semantic prototype
        matrix for the test set.

        :return: 2D numpy array with semantic prototype for each instance in the test set
        """
        lbs = {self.data['te_cl_id'][j][0]: attributes for j, attributes in enumerate(self.data['S_te_pro'])}
        return np.array([lbs[label[0]] for label in self.data['test_labels_cub']], dtype=np.float64)

    def _get_te_pro_labels(self):
        """
        Parses the labels in the prototype matrix. In this case it is important to keep the order of the labels
        for correct mapping later.

        :return: 1D numpy array with the labels for each semantic prototype in S_te_pro matrix
        """
        return np.array([int(x) for x in self.data['te_cl_id']])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parses datasets saved in .mat and saves the extracted data into a '
                                                 'compressed pickle file. Inputs are the datasets created by Elyor '
                                                 'Kodirov, Tao Xiang, and Shaogang Gong in the paper "Semantic '
                                                 'Autoencoder for Zero-shot Learning" published in CVPR 2017.')

    parser.add_argument('--input', help='Name and path of the input file')
    parser.add_argument('--output', help='Name of the output file')
    parser.add_argument('--folder', help='Path of the output file', default='./data')
    parser.add_argument('--type', help='Type of dataset to be parsed', default='awa', choices=['awa', 'cub'])

    args = parser.parse_args()

    if args.type == 'awa':
        AWAParser(args.input)(args.folder, args.output)
    elif args.type == 'cub':
        CUBParser(args.input)(args.folder, args.output)
    else:
        raise ValueError('Invalid data type.')
