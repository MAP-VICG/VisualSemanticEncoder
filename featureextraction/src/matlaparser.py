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
from scipy.io import savemat


class Parser:
    def __init__(self, data_type):
        """
        Sets test labels and data type

        :param data_type: type of data analyzed: awa or cub.
        """
        self.data_type = data_type
        self.data_dict = dict()

        if self.data_type == 'awa':
            self.test_labels = [25, 39, 15,  6, 42, 14, 18, 48, 34, 24]
        elif self.data_type == 'cub':
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

        if self.data_type == 'awa':
            self.data_dict['S_te_gt'] = self.build_semantic_matrix(sem_data, labels)
            self.data_dict['param'] = dict()
            self.data_dict['param']['testclasses_id'] = np.expand_dims(np.array(self.test_labels), axis=1)
            self.data_dict['param']['test_labels'] = np.expand_dims(np.array(labels[te_mask]), axis=1)
            self.data_dict['param']['train_labels'] = np.expand_dims(np.array(labels[tr_mask]), axis=1)

        elif self.data_type == 'cub':
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

    @staticmethod
    def load_awa_data(vis_data_path, sem_data_path, labels_data_path):
        """
        Loads AwA data into 2D numpy arrays for visual data, semantic data and labels

        :param vis_data_path: path to visual data
        :param sem_data_path: path to semantic data
        :param labels_data_path: path to labels file
        :return: tuple with visual data, semantic data and labels
        """
        with open(sem_data_path) as f:
            sem_data = [list(map(float, line.split(' '))) for line in f.readlines()]

        with open(vis_data_path) as f:
            vis_data = [list(map(float, line.split(' '))) for line in f.readlines()]

        with open(labels_data_path) as f:
            labels = [int(line.strip()) for line in f.readlines()]

        return np.array(vis_data), np.array(sem_data), np.array(labels)

    @staticmethod
    def load_cub_data(tr_vis_path, te_vis_path, tr_sem_path, te_sem_path, tr_labels_path, te_labels_path):
        """
        Loads CUB200 data into 2D numpy arrays for visual data, semantic data and labels

        :param tr_vis_path: path to visual training data
        :param te_vis_path: path to visual test data
        :param tr_sem_path: path to semantic training data
        :param te_sem_path: path to semantic test data
        :param tr_labels_path: path to training labels file
        :param te_labels_path: path to test labels file
        :return: tuple with visual data, semantic data and labels
        """
        with open(tr_sem_path) as f:
            sem_data = [list(map(float, line.split(' '))) for line in f.readlines()]

        with open(te_sem_path) as f:
            sem_data.extend([list(map(float, line.split(' '))) for line in f.readlines()])

        with open(tr_vis_path) as f:
            vis_data = [list(map(float, line.split(' '))) for line in f.readlines()]

        with open(te_vis_path) as f:
            vis_data.extend([list(map(float, line.split(' '))) for line in f.readlines()])

        with open(tr_labels_path) as f:
            labels = [int(line.strip()) for line in f.readlines()]

        with open(te_labels_path) as f:
            labels.extend([int(line.strip()) for line in f.readlines()])

        return np.array(vis_data), np.array(sem_data), np.array(labels)


if __name__ == '__main__':
    awa_prs = Parser('awa')
    x_vis = '../../../Datasets/AWA2/AWA2_x_vis.txt'
    x_sem = '../../../Datasets/AWA2/AWA2_x_sem.txt'
    y = '../../../Datasets/AWA2/AWA2_y.txt'
    awa_vis_data, awa_sem_data, awa_labels = awa_prs.load_awa_data(x_vis, x_sem, y)
    awa_prs.split_data(awa_vis_data, awa_sem_data, awa_labels)
    awa_prs.save_data('../../../Datasets/SAE/awa_demo_data_resnet.mat')

    cub_prs = Parser('cub')
    tr_vis = '../../../Datasets/CUB200/CUB200_x_train_vis.txt'
    te_vis = '../../../Datasets/CUB200/CUB200_x_test_vis.txt'
    tr_sem = '../../../Datasets/CUB200/CUB200_x_train_sem.txt'
    te_sem = '../../../Datasets/CUB200/CUB200_x_test_sem.txt'
    tr_y = '../../../Datasets/CUB200/CUB200_y_train.txt'
    te_y = '../../../Datasets/CUB200/CUB200_y_test.txt'
    cub_vis_data, cub_sem_data, cub_labels = cub_prs.load_cub_data(tr_vis, te_vis, tr_sem, te_sem, tr_y, te_y)
    cub_prs.split_data(cub_vis_data, cub_sem_data, cub_labels)
    cub_prs.save_data('../../../Datasets/SAE/cub_demo_data_resnet.mat')
