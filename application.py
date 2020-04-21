"""
Encodes visual and semantic features of images

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Mar 23, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""
import time
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import normalize

from encoding.src.plotter import Plotter
from encoding.src.encoder import ModelFactory, ModelType


def main():
    init_time = time.time()
    data = 'cub'

    if data == 'awa':
        sem_length = 312
        baseline = 0.614
        cub = loadmat('../Datasets/SAE/cub_demo_data.mat')
        x_train_vis = cub['X_tr']
        x_train_sem = normalize(cub['S_tr'], norm='l2', axis=1, copy=True)

        x_test_vis = cub['X_te']
        labels_map = cub['S_te_pro']

        testclasses_id = np.array([int(x) for x in cub['te_cl_id']])
        test_labels = np.array([int(x) for x in cub['test_labels_cub']])
    else:
        sem_length = 85
        baseline = 0.847
        awa = loadmat('../Datasets/SAE/awa_demo_data.mat')
        x_train_vis = awa['X_tr']
        x_train_sem = normalize(awa['S_tr'], norm='l2', axis=1, copy=True)

        x_test_vis = awa['X_te']
        labels_map = awa['S_te_gt']

        testclasses_id = np.array([int(x) for x in awa['param']['testclasses_id'][0][0]])
        test_labels = np.array([int(x) for x in awa['param']['test_labels'][0][0]])

    mapping = {value: idx for idx, value in enumerate(testclasses_id)}
    x_test_sem = np.array([labels_map[mapping[label], :] for label in test_labels])

    ae = ModelFactory(1024, sem_length, sem_length)(ModelType.SEMANTIC_VAE)
    ae.run_model(vis_fts=x_train_vis, sem_fts=x_train_sem, num_epochs=5, x_test_vis=x_test_vis,
                 x_test_sem=x_test_sem, sem_length=sem_length,
                 labels_map=labels_map, testclasses_id=testclasses_id, test_labels=test_labels)

    pt = Plotter(ae, './_files/')
    pt.plot_evaluation(baseline)
    print(max(ae.zsl_accuracies))

    elapsed = time.time() - init_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    time_elapsed = '{:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds)

    print('Elapsed time is %s' % time_elapsed)
    
    
if __name__ == '__main__':
    main()
