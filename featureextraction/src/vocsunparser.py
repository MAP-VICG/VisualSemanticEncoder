from os import path
import numpy as np
from scipy.io import loadmat, savemat


def build_sun_data():
    with open('../SUN_x_train_sem.txt') as f:
        sun_sem = np.array([list(map(float, line.split())) for line in f.readlines()])

    with open('../SUN_x_train_vis.txt') as f:
        sun_vis = np.array([list(map(float, line.split())) for line in f.readlines()])

    with open('../SUN_y_train.txt') as f:
        sun_labels = np.array([list(map(float, line.split())) for line in f.readlines()])

    data = loadmat('/store/shared/datasets/SUNAttibutes/SUNAttributeDB/images.mat')
    labels = [image[0][0].split('/')[1] for image in data['images']]

    data_structure = dict()
    data_structure['S_tr'] = sun_sem
    data_structure['X_tr'] = sun_vis
    data_structure['S_te_pro'] = sun_sem
    data_structure['tr_cl_id'] = list(set([int(lb[0]) for lb in sun_labels]))
    data_structure['train_labels'] = np.array([[int(lb[0])] for lb in sun_labels])
    data_structure['labels_dict'] = {label.strip(): i + 1 for i, label in enumerate(set(labels))}
    data_structure['X_te'] = data_structure['S_tr'] = data_structure['test_labels'] = data_structure['te_cl_id'] = 'none'

    savemat('../sun_data_resnet50.mat', data_structure)


def build_voc_data():
    with open('../aP&Y_x_train_sem.txt') as f:
        voc_sem = np.array([list(map(float, line.split())) for line in f.readlines()])

    with open('../aP&Y_x_train_vis.txt') as f:
        voc_vis = np.array([list(map(float, line.split())) for line in f.readlines()])

    with open('../aP&Y_y_train.txt') as f:
        voc_labels = np.array([list(map(float, line.split())) for line in f.readlines()])

    images_path = 'store/shared/datasets/aPascalYahoo/images'
    with open('/store/shared/datasets/aPascalYahoo/attribute_data/apascal_train.txt') as f:
        images_list = [path.join('VOC2012', 'trainval', 'JPEGImages', line.split()[0]) for line in f.readlines()]

    with open('/store/shared/datasets/aPascalYahoo/attribute_data/apascal_test.txt') as f:
        images_list.extend([path.join('VOC2012', 'test', 'JPEGImages', line.split()[0]) for line in f.readlines()])

    with open('/store/shared/datasets/aPascalYahoo/attribute_data/ayahoo_test.txt') as f:
        images_list.extend([path.join('Yahoo', line.split()[0]) for line in f.readlines()])

    with open('/store/shared/datasets/aPascalYahoo/attribute_data/ayahoo_test.txt') as f:
        limit = sum([1 for img in f.readlines() if path.isfile(path.join(images_path, 'Yahoo', img.split()[0]))])

    images_mask = [True if path.isfile(path.join(images_path, img)) else False for img in images_list]
    images_list = np.array(images_list)[images_mask]

    with open('/store/shared/datasets/aPascalYahoo/attribute_data/class_names.txt') as f:
        labels_dict = {label.strip(): i + 1 for i, label in enumerate(f.readlines()) if images_mask[i]}

    data_structure = dict()
    data_structure['S_tr'] = voc_sem[:len(images_list) - limit, :]
    data_structure['X_tr'] = voc_vis[:len(images_list) - limit, :]
    data_structure['S_te_pro'] = voc_sem[limit:, :]
    data_structure['tr_cl_id'] = list(set([int(lb[0]) for lb in voc_labels[:len(images_list) - limit, :]]))
    data_structure['train_labels'] = np.array([[int(lb[0])] for lb in voc_labels[:len(images_list) - limit, :]])
    data_structure['labels_dict'] = labels_dict
    data_structure['X_te'] = voc_vis[limit:, :]
    data_structure['S_tr'] = voc_sem[limit:, :]
    data_structure['test_labels'] = np.array([[int(lb[0])] for lb in voc_labels[limit:, :]])
    data_structure['te_cl_id'] = list(set([int(lb[0]) for lb in voc_labels[limit:, :]]))

    savemat('../apy_data_resnet50.mat', data_structure)


build_sun_data()
build_voc_data()
