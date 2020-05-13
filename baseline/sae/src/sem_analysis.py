import json
import random
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import normalize

from baseline.sae.src.utils import ZSL


class SemanticDegradation:
    def __init__(self, datafile, data_type, new_value=None):
        """

        :param data_type:
        :param new_value: real value to replace to. If not specified, a random value will be chosen
        """
        self.data_type = data_type
        self.new_value = new_value
        self.data = loadmat(datafile)

        if data_type == 'awa':
            self.z_score = False
            self.temp_labels = np.array([int(x) for x in self.data['param']['testclasses_id'][0][0]])
            self.test_labels = np.array([int(x) for x in self.data['param']['test_labels'][0][0]])
            self.s_te_pro = normalize(self.data['S_te_pro'].transpose(), norm='l2', axis=1, copy=True).transpose()
        elif data_type == 'cub':
            self.z_score = True
            self.temp_labels = np.array([int(x) for x in self.data['te_cl_id']])
            self.test_labels = np.array([int(x) for x in self.data['test_labels_cub']])
            self.s_te_pro = self.data['S_te_pro']

            labels = list(map(int, self.data['train_labels_cub']))
            self.data['X_tr'], self.data['X_te'] = ZSL.dimension_reduction(self.data['X_tr'], self.data['X_te'], labels)

        if new_value is None:
            self.limits = (np.min(self.data['S_tr']), np.max(self.data['S_tr']))

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

    def degrade_semantic_data(self, n_folds):
        """
        Randomly replaces the values of the semantic array for a new value specified and runs SAE over it.
        Saves the resultant accuracy in a dictionary. Data is degraded with rates ranging from 10 to 100%
        for a specific number of folds.

        :return: None
        """
        rates = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        acc_dict = {key: {'acc': np.zeros(n_folds), 'mean': 0, 'std': 0, 'max': 0, 'min': 0} for key in rates}

        for rate in rates:
            for j in range(n_folds):
                s_tr = self.kill_semantic_attributes(self.data['S_tr'], rate)
                s_te = self.estimate_semantic_data(self.data['X_tr'], s_tr, self.data['X_te'])

                acc, _ = ZSL.zsl_el(s_te, self.s_te_pro, self.test_labels, self.temp_labels, 1, self.z_score)
                acc_dict[rate]['acc'][j] = acc

            acc_dict[rate]['mean'] = np.mean(acc_dict[rate]['acc'])
            acc_dict[rate]['std'] = np.std(acc_dict[rate]['acc'])
            acc_dict[rate]['max'] = np.max(acc_dict[rate]['acc'])
            acc_dict[rate]['min'] = np.min(acc_dict[rate]['acc'])
            acc_dict[rate]['acc'] = ', '.join(list(map(str, acc_dict[rate]['acc'])))

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


if __name__ == '__main__':
    sem = SemanticDegradation('../../../../Datasets/SAE/awa_demo_data.mat', 'awa')
    sem.write2json(sem.degrade_semantic_data(n_folds=10), '../../../plotter/data/awa_v2s_projection_random.json')

    sem = SemanticDegradation('../../../../Datasets/SAE/cub_demo_data.mat', 'cub')
    sem.write2json(sem.degrade_semantic_data(n_folds=10), '../../../plotter/data/cub_v2s_projection_random.json')
