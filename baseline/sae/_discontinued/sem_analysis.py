"""
Runs SAE for different sets of semantic data. These data is modified according to a rate
and a value to set. Rates from 10% to 100% are tested. As well as well as different values
such as 0, 0.001, 0.1, -1 and 9999.

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Apr 27, 2020

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""
import json
import random
import logging
import numpy as np

from baseline.sae.src.awa_demo import AWA
from baseline.sae.src.cub_demo import CUB200


class SemanticDegradation:
    def __init__(self, datafile, n_folds, new_value, data_type, verbose=False):
        """
        Initializes auxiliary variables.

        :param datafile: .mat file with data
        :param n_folds: number of folds to degrade data per rate
        :param new_value: value to replace the original one
        :param data_type: type of data to be analysed ("cub" or "awa")
        :param verbose: if true, activates printing messages to log file
        """
        if data_type == 'awa':
            self.data = AWA(datafile)
        elif data_type == 'cub':
            self.data = CUB200(datafile)

        self.n_folds = n_folds
        self.verbose = verbose
        self.rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

        if isinstance(new_value, int):
            self.limits = None
            self.new_value = new_value
        elif isinstance(new_value, str) and new_value == 'random':
            self.new_value = None
            self.limits = (np.min(self.data.data['S_tr']), np.max(self.data.data['S_tr']))
        else:
            raise ValueError('Unexpected value for new_value. Please choose a real number or the string "random"')

        self.acc = {key: {'acc': np.zeros(self.n_folds), 'mean': 0, 'std': 0, 'max': 0, 'min': 0} for key in self.rates}

        self.data.set_semantic_data()
        self.acc['ref'] = self.data.v2s_projection()
        self.acc['new_value'] = new_value

        if self.verbose:
            self.print_degradation_status(0)

    def print_degradation_status(self, rate):
        """
        Prints the number of values that could be modified in the semantic array.

        :param rate: rate expected in degradation
        :return: None
        """
        counts = [0] * self.data.s_tr.shape[0]
        for i in range(self.data.s_tr.shape[0]):
            for j in range(self.data.s_tr.shape[1]):
                if self.data.s_tr[i][j] == self.new_value:
                    counts[i] += 1
        logger.info('Number of degraded values is %s for rate %.2f', str(set(counts)), rate)

    def degrade_semantic_data(self):
        """
        Randomly replaces the values of the semantic array for a new value specified and runs SAE over it.
        Saves the resultant accuracy in a dictionary. Data is degraded with rates ranging from 10 to 100%
        for a specific number of folds.

        :return: None
        """
        for rate in self.rates:
            for j in range(self.n_folds):
                self.data.reset_weights()
                self.data.set_semantic_data(self.kill_semantic_attributes(self.data.data['S_tr'], rate))

                if self.verbose:
                    logger.info('Computing fold %d...', j)
                    self.print_degradation_status(rate)

                self.acc[rate]['acc'][j] = self.data.v2s_projection()

            self.acc[rate]['mean'] = np.mean(self.acc[rate]['acc'])
            self.acc[rate]['std'] = np.std(self.acc[rate]['acc'])
            self.acc[rate]['max'] = np.max(self.acc[rate]['acc'])
            self.acc[rate]['min'] = np.min(self.acc[rate]['acc'])
            self.acc[rate]['acc'] = ', '.join(list(map(str, self.acc[rate]['acc'])))

    def write2json(self, filename):
        """
        Writes data from accuracy dictionary to JSON file

        :param filename: string with name of file to write data to
        :return: None
        """
        json_string = json.dumps(self.acc)
        with open(filename, 'w+') as f:
            json.dump(json_string, f)

    def kill_semantic_attributes(self, data, rate):
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

            if self.limits is None and self.new_value is not None:
                new_data[ex, mask] = new_data[ex, mask] * 0 + self.new_value
            else:
                for i in range(len(mask)):
                    if mask[i]:
                        new_data[ex, i] = random.uniform(self.limits[0], self.limits[1])

        return new_data


if __name__ == '__main__':
    FORMAT = '%(asctime)-15s %(message)s'
    FILENAME = '../../../plotter/data/semantic_degradation.log'
    logging.basicConfig(filename=FILENAME, format=FORMAT, filemode='a', level=logging.DEBUG)
    logger = logging.getLogger('SemanticDegradation')

    for key_value in ['random', 0, 0.001, 0.1, -1, 9999]:
        logger.info('Key value is set to %s', str(key_value))

        logger.info('Evaluating AWA')
        sem = SemanticDegradation('../../../../Datasets/SAE/awa_demo_data.mat', 10, key_value, 'awa', verbose=True)
        sem.degrade_semantic_data()
        sem.write2json('../../../plotter/data/awa_v2s_projection_%s.json' % str(key_value))

        logger.info('Evaluating CUB200')
        sem = SemanticDegradation('../../../../Datasets/SAE/cub_demo_data.mat', 10, key_value, 'cub', verbose=True)
        sem.degrade_semantic_data()
        sem.write2json('../../../plotter/data/cub_v2s_projection_%s.json' % str(key_value))
