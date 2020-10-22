"""
Retrieves basic information about the CUB200, AwA2, aPascalYahoo,
and SUN attributes datasets.

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Jan 5, 2020

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""
import logging
import numpy as np
from os import path, listdir
from scipy.io import loadmat, savemat

from .fetureextraction import ExtractorFactory, ExtractionType


class DataParser:
    def __init__(self, base_path, extractor):
        if not isinstance(extractor, ExtractionType):
            raise ValueError('Invalid extractor type')

        self.extractor = extractor
        self.base_path = self.classes_path = self.images_path = self.semantic_attributes_path = base_path

    def build_data_structure(self, file_name, vis_data=True):
        data = dict()
        logging.info('Extracting AwA data. Using extractor %s.' % self.extractor.name)
        data['img_list'], data['img_class'], data['class_dict'] = self.get_images_data()
        data['class_dict'] = [(key, value) for key, value in data['class_dict'].items()]

        logging.info('Images length: %d. Classes length: %d.' % (len(data['img_list']), len(data['img_class'])))
        logging.info('Number of classes: %d' % len(data['class_dict']))

        data['sem_fts'], data['prototypes'] = self.get_semantic_attributes(data['img_class'])
        logging.info('Semantic features shape: %s' % str(data['sem_fts'].shape))

        if not vis_data:
            logging.warning('Visual data will not be extracted')
        else:
            data['vis_fts'] = self.get_visual_attributes(data['img_list'])
            logging.info('Visual features shape: %s' % str(data['vis_fts'].shape))

        savemat(file_name, data)

    def get_images_data(self):
        return None, None, None

    def get_semantic_attributes(self, img_class_list):
        try:
            with open(self.semantic_attributes_path) as f:
                attributes = [list(map(float, line.split())) for line in f.readlines()]

            sem_attributes = list()
            for klass in img_class_list:
                sem_attributes.append(attributes[klass - 1])

            return np.array(sem_attributes), np.array(attributes)
        except (IOError, FileNotFoundError):
            return None, None

    def get_visual_attributes(self, images_list):
        ext = ExtractorFactory(self.images_path)(self.extractor)
        return ext.extract_images_list_features(images_list)


class AWA2Data(DataParser):
    def __init__(self, base_path, extractor):
        super(AWA2Data, self).__init__(base_path, extractor)
        self.images_path = path.join(base_path, 'JPEGImages')
        self.semantic_attributes_path = path.join(base_path, 'predicate-matrix-continuous.txt')

    def get_images_data(self):
        img_list = list()
        img_class = list()

        with open(path.join(self.base_path, 'classes.txt')) as f:
            class_dict = {name.split()[1]: int(name.split()[0]) for name in f.readlines()}

        for folder in listdir(self.images_path):
            if path.isdir(path.join(self.images_path, folder)):
                for img in listdir(path.join(self.images_path, folder)):
                    if img.endswith('.jpg'):
                        img_list.append(path.join(folder, img))
                        img_class.append(class_dict[img.split('_')[0]])

        return img_list, img_class, class_dict


class CUB200Data(DataParser):
    def __init__(self, base_path, extractor):
        super(CUB200Data, self).__init__(base_path, extractor)
        self.images_path = path.join(base_path, 'images')
        self.semantic_attributes_path = path.join(base_path, 'attributes', 'class_attribute_labels_continuous.txt')

    def get_images_data(self):
        img_list = list()
        class_dict = dict()

        with open(path.join(self.base_path, 'images.txt')) as f:
            for value in f.readlines():
                img_list.append(value.split()[1])
                idx, name = img_list[-1].split('/')[0].split('.')
                class_dict[name.strip()] = int(idx.strip())

        with open(path.join(self.base_path, 'image_class_labels.txt')) as f:
            img_class = [int(value.split()[1]) for value in f.readlines()]

        return img_list, img_class, class_dict


class PascalYahooData(DataParser):
    def __init__(self, base_path, extractor):
        super(PascalYahooData, self).__init__(base_path, extractor)
        self.semantic_data = None
        self.images_path = path.join(base_path, 'images')
        self.semantic_attributes_path = path.join(base_path, 'attribute_data')

    def get_images_data(self):
        img_list = list()
        img_class = list()
        semantic_data = list()

        with open(path.join(self.semantic_attributes_path, 'class_names.txt')) as f:
            class_dict = {label.strip(): i + 1 for i, label in enumerate(f.readlines())}

        def _build_lists(lines, base_path):
            for line in lines:
                values = line.split()
                if path.isfile(path.join(self.images_path, base_path, values[0].strip())):
                    img_list.append(path.join(base_path, values[0].strip()))
                    img_class.append(int(class_dict[values[1].strip()]))
                    semantic_data.append(values[6:])

        with open(path.join(self.semantic_attributes_path, 'apascal_train.txt')) as f:
            _build_lists(f.readlines(), path.join('VOC2012', 'train', 'JPEGImages'))

        with open(path.join(self.semantic_attributes_path, 'apascal_test.txt')) as f:
            _build_lists(f.readlines(), path.join('VOC2012', 'test', 'JPEGImages'))

        with open(path.join(self.semantic_attributes_path, 'ayahoo_test.txt')) as f:
            _build_lists(f.readlines(), path.join('Yahoo'))

        self.semantic_data = np.array(semantic_data)
        return img_list, img_class, class_dict

    def get_semantic_attributes(self, img_class):
        return self.semantic_data, self.semantic_data


class SUNData(DataParser):
    def __init__(self, base_path, extractor):
        super(SUNData, self).__init__(base_path, extractor)
        self.images_path = path.join(base_path, 'images')
        self.semantic_attributes_path = path.join(base_path, 'SUNAttributeDB')

    def get_images_data(self):
        data = loadmat(path.join(self.semantic_attributes_path, 'images.mat'))
        img_list = np.array([image[0][0] for image in data['images']])
        img_class_names = ['_'.join(image.split('/')[:-1]) for image in img_list]

        class_dict = {name: i + 1 for i, name in enumerate(set(img_class_names))}
        img_class = [class_dict[img] for img in img_class_names]

        return img_list, img_class, class_dict

    def get_semantic_attributes(self, img_class):
        data = loadmat(path.join(self.semantic_attributes_path, 'attributeLabels_continuous.mat'))
        return data['labels_cv']
