import numpy as np
from scipy.io import loadmat, savemat

def build_sun_data():
    with open('../../../../Desktop/SUN_x_train_sem.txt') as f:
        sun_sem = np.array([list(map(float, line.split())) for line in f.readlines()])

    with open('../../../../Desktop/SUN_x_train_vis.txt') as f:
        sun_vis = np.array([list(map(float, line.split())) for line in f.readlines()])

    with open('../../../../Desktop/SUN_y_train.txt') as f:
        sun_labels = np.array([list(map(float, line.split())) for line in f.readlines()])

    data = loadmat('../../../../Downloads/SUNAttributeDB/images.mat')
    labels = [image[0][0].split('/')[1] for image in data['images']]

    data_structure = dict()
    data_structure['S_tr'] = sun_sem
    data_structure['X_tr'] = sun_vis
    data_structure['S_te_pro'] = sun_sem
    data_structure['tr_cl_id'] = list(set([int(lb[0]) for lb in sun_labels]))
    data_structure['train_labels'] = np.array([[int(lb[0])] for lb in sun_labels])
    data_structure['labels_dict'] = {label.strip(): i + 1 for i, label in enumerate(set(labels))}
    data_structure['X_te'] = data_structure['S_tr'] = data_structure['test_labels'] = data_structure['te_cl_id'] = 'none'

    savemat('../../../Datasets/sun_data_resnet50.mat', data_structure)


def build_voc_data():
    with open('../../../../Desktop/aP&Y_x_train_sem.txt') as f:
        voc_sem = np.array([list(map(float, line.split())) for line in f.readlines()])

    with open('../../../../Desktop/aP&Y_x_train_vis.txt') as f:
        voc_vis = np.array([list(map(float, line.split())) for line in f.readlines()])

    with open('../../../../Desktop/aP&Y_y_train.txt') as f:
        voc_labels = np.array([list(map(float, line.split())) for line in f.readlines()])

    pass


build_voc_data()
