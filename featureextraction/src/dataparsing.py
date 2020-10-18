"""
Retrieves basic information about the CUB200 and AwA2 data sets

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Jan 5, 2020

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""
import os
import numpy as np
from scipy.io import loadmat
from os import path, listdir, sep

from featureextraction.src.fetureextraction import ResNet50FeatureExtractor, InceptionV3FeatureExtractor


class DataParser:
    def __init__(self, base_path, extractor):
        if extractor not in ('resnet', 'inception'):
            raise Exception('Invalid extractor for visual features')

        self.extractor = extractor
        self.images_path = ''
        self.images_list = None
        self.base_path = base_path
        self.semantic_attributes_path = ''

    def get_semantic_attributes(self):
        """
        Reads the file of attributes and retrieves the semantic array for each class

        @return: float numpy array of shape (X, Y) being X the number of classes and Y the number of attributes
        """
        try:
            with open(self.semantic_attributes_path) as f:
                attributes = [list(map(float, line.split())) for line in f.readlines()]
            return np.array(attributes)
        except (IOError, FileNotFoundError):
            return None

    def get_visual_attributes(self, images_list):
        """
        Extracts the visual features of every image listed using ResNet50 model

        @param images_list: list of strings with image's names
        @return: numpy array of shape (X, 2048) where X is the number of images listed
        """
        if self.extractor == 'resnet':
            vis_data = ResNet50FeatureExtractor(images_list, self.images_path)
        else:
            vis_data = InceptionV3FeatureExtractor(images_list, self.images_path)

        vis_data.extract_images_list_features()
        return vis_data.features_set

    def _get_images_list(self):
        """
        Retrieves path of each image

        @return: integer 1D numpy array
        """
        return None

    def get_images_list(self):
        """
        Wrapper method to retrieve path of each image

        @return: integer 1D numpy array
        """
        if self.images_list is None:
            self.images_list = self._get_images_list()
        return self.images_list

    def get_images_class(self):
        """
        Retrieves the class of each image

        @return: integer 1D numpy array
        """
        return None

    def build_data(self):
        """
        Builds a data set with visual features extracted from ResNet50 and semantic features extracted from labels file

        @return: tuple with visual features, semantic features and classes
        """
        images = self.get_images_list()
        classes = self.get_images_class()

        sem_msk = self.get_semantic_attributes()
        vis_fts = self.get_visual_attributes(images)

        sem_fts = []
        for i, fts in enumerate(vis_fts):
            sem_fts.append(sem_msk[classes[i] - 1, :])

        return vis_fts, np.array(sem_fts), classes


class AWA2Data(DataParser):
    def __init__(self, base_path, extractor):
        super(AWA2Data, self).__init__(base_path, extractor)
        self.images_path = path.join(base_path, 'JPEGImages')
        self.semantic_attributes_path = path.join(base_path, 'predicate-matrix-continuous.txt')
        self.images_list = None

    def _get_images_list(self):
        """
        Retrieves path of each image
        @return: integer 1D numpy array
        """
        images_list = [path.join(folder, img) for folder in listdir(self.images_path)
                       if path.isdir(path.join(self.images_path, folder))
                       for img in listdir(path.join(self.images_path, folder)) if img.endswith('.jpg')]
        return np.array(sorted(images_list))

    def get_images_class(self):
        """
        Retrieves the class of each image

        @return: integer 1D numpy array
        """
        with open(path.join(self.base_path, 'classes.txt')) as f:
            labels_dict = {label.split()[1]: int(label.split()[0]) for label in f.readlines()}

        labels = [labels_dict[img.strip().split(sep)[0]] for img in self.get_images_list()]

        return np.array(labels)


class CUB200Data(DataParser):
    def __init__(self, base_path, extractor):
        super(CUB200Data, self).__init__(base_path, extractor)
        self.images_path = path.join(base_path, 'images')
        self.semantic_attributes_path = path.join(base_path, 'attributes', 'class_attribute_labels_continuous.txt')

    def _get_images_list(self):
        """
        Retrieves path of each image

        @return: integer 1D numpy array
        """
        with open(path.join(self.base_path, 'images.txt')) as f:
            images_list = [value.split()[1] for value in f.readlines()]

        return np.array(images_list)

    def get_images_class(self):
        """
        Retrieves the class of each image

        @return: integer 1D numpy array
        """
        with open(path.join(self.base_path, 'image_class_labels.txt')) as f:
            labels = [int(value.split()[1]) for value in f.readlines()]

        return np.array(labels)

    def get_train_test_mask(self):
        """
        Goes through the full list of images and creates a boolean mask to split data set into training and test sets,
        where True indicates training set and False test set. Then builds boolean arrays to mask data sets.

        @return: tuple with boolean arrays for training and test masks
        """

        with open(path.join(self.base_path, 'train_test_split.txt')) as f:
            mask = [int(value.split()[1]) for value in f.readlines()]

        return np.array(mask) == 1, np.array(mask) == 0


class PascalYahooData(DataParser):
    def __init__(self, base_path, extractor):
        super(PascalYahooData, self).__init__(base_path, extractor)
        self.images_path = path.join(base_path, 'images')
        self.semantic_attributes_path = path.join(base_path, 'attribute_data')

    def _get_images_list(self):
        """
        Retrieves path of each image

        @return: integer 1D numpy array
        """
        with open(path.join(self.semantic_attributes_path, 'apascal_train.txt')) as f:
            images_list = [path.join('VOC2012', 'trainval', 'JPEGImages', line.split()[0]) for line in f.readlines()]

        with open(path.join(self.semantic_attributes_path, 'apascal_test.txt')) as f:
            images_list.extend([path.join('VOC2012', 'test', 'JPEGImages', line.split()[0]) for line in f.readlines()])

        with open(path.join(self.semantic_attributes_path, 'ayahoo_test.txt')) as f:
            images_list.extend([path.join('Yahoo', line.split()[0]) for line in f.readlines()])

        return np.array([img for img in images_list if os.path.isfile(path.join(self.images_path, img))])

    def get_images_class(self):
        """
        Retrieves the class of each image

        @return: integer 1D numpy array
        """
        labels = []
        with open(path.join(self.semantic_attributes_path, 'class_names.txt')) as f:
            labels_dict = {label.strip(): i + 1 for i, label in enumerate(f.readlines())}

        with open(path.join(self.semantic_attributes_path, 'apascal_train.txt')) as f:
            for line in f.readlines():
                if os.path.isfile(path.join(self.images_path, 'VOC2012', 'trainval', 'JPEGImages', line.split()[0])):
                    labels.append(int(labels_dict[line.split()[1].strip()]))

        with open(path.join(self.semantic_attributes_path, 'apascal_test.txt')) as f:
            for line in f.readlines():
                if os.path.isfile(path.join(self.images_path, 'VOC2012', 'test', 'JPEGImages', line.split()[0])):
                    labels.append(int(labels_dict[line.split()[1].strip()]))

        with open(path.join(self.semantic_attributes_path, 'ayahoo_test.txt')) as f:
            for line in f.readlines():
                if os.path.isfile(path.join(self.images_path, 'Yahoo', line.split()[0])):
                    labels.append(int(labels_dict[line.split()[1].strip()]))

        return np.array(labels)

    def get_semantic_attributes(self):
        """
        Reads the file of attributes and retrieves the semantic array for each class

        @return: float numpy array of shape (X, Y) being X the number of classes and Y the number of attributes
        """
        try:
            attributes = []
            with open(path.join(self.semantic_attributes_path, 'apascal_train.txt')) as f:
                for line in f.readlines():
                    if os.path.isfile(path.join(self.images_path, 'VOC2012', 'trainval', 'JPEGImages', line.split()[0])):
                        attributes.append(list(map(float, line.split()[6:])))

            with open(path.join(self.semantic_attributes_path, 'apascal_test.txt')) as f:
                for line in f.readlines():
                    if os.path.isfile(path.join(self.images_path, 'VOC2012', 'test', 'JPEGImages', line.split()[0])):
                        attributes.append(list(map(float, line.split()[6:])))

            with open(path.join(self.semantic_attributes_path, 'ayahoo_test.txt')) as f:
                for line in f.readlines():
                    if os.path.isfile(path.join(self.images_path, 'Yahoo', line.split()[0])):
                        attributes.append(list(map(float, line.split()[6:])))

            return np.array(attributes)
        except (IOError, FileNotFoundError):
            return None

    def build_data(self):
        """
        Builds a data set with visual features extracted from ResNet50 and semantic features extracted from labels file

        @return: tuple with visual features, semantic features and classes
        """
        images = self.get_images_list()
        classes = self.get_images_class()

        sem_fts = self.get_semantic_attributes()
        vis_fts = self.get_visual_attributes(images)

        return vis_fts, sem_fts, classes


class SUNData(DataParser):
    def __init__(self, base_path, extractor):
        super(SUNData, self).__init__(base_path, extractor)
        self.images_path = path.join(base_path, 'images')
        self.semantic_attributes_path = path.join(base_path, 'SUNAttributeDB')

    def _get_images_list(self):
        """
        Retrieves path of each image

        @return: integer 1D numpy array
        """
        data = loadmat(path.join(self.semantic_attributes_path, 'images.mat'))
        return np.array([image[0][0] for image in data['images']])

    def get_images_class(self):
        """
        Retrieves the class of each image

        @return: integer 1D numpy array
        """
        data = loadmat(path.join(self.semantic_attributes_path, 'images.mat'))
        labels = ['_'.join(image[0][0].split('/')[:-1]) for image in data['images']]
        labels_dict = {label.strip(): i + 1 for i, label in enumerate(set(labels))}

        return np.array([labels_dict[lb] for lb in labels])

    def get_semantic_attributes(self):
        """
        Reads the file of attributes and retrieves the semantic array for each class

        @return: float numpy array of shape (X, Y) being X the number of classes and Y the number of attributes
        """
        data = loadmat(path.join(self.semantic_attributes_path, 'attributeLabels_continuous.mat'))
        return data['labels_cv']

    def build_data(self):
        """
        Builds a data set with visual features extracted from ResNet50 and semantic features extracted from labels file

        @return: tuple with visual features, semantic features and classes
        """
        images = self.get_images_list()
        classes = self.get_images_class()

        sem_fts = self.get_semantic_attributes()
        vis_fts = self.get_visual_attributes(images)

        return vis_fts, sem_fts, classes


class DataIO:
    @staticmethod
    def get_features(data_file):
        """
        Builds a 2D numpy array with data found in file

        @param data_file: string with full path to data file
        @return: float numpy array of shape (X, Y) where X is the number of images and Y the number of attributes
        """
        with open(data_file) as f:
            data = [list(map(float, line.split())) for line in f.readlines()]
        return np.array(data)

    @staticmethod
    def get_labels(labels_file):
        """
        Builds a 1D numpy array with data found in file

        @param labels_file: string with full path to labels file
        @return: integer numpy array of shape (X, ) where X is the number of images
        """
        with open(labels_file) as f:
            labels = list(map(int, f.readlines()))
        return np.array(labels)

    @staticmethod
    def save_files(base_path, prefix, **kwargs):
        """
        Saves sets data into files

        @param base_path: string with base path to save files
        @param prefix: string with prefix to identify data set
        @param kwargs: data sets
        @return: None
        """
        if 'x_train_vis' in kwargs.keys():
            with open(path.join(base_path, '%s_x_train_vis.txt' % prefix), 'w+') as f:
                for instance in kwargs['x_train_vis']:
                    f.write(' '.join(list(map(str, instance))) + '\n')

        if 'x_train_sem' in kwargs.keys():
            with open(path.join(base_path, '%s_x_train_sem.txt' % prefix), 'w+') as f:
                for instance in kwargs['x_train_sem']:
                    f.write(' '.join(list(map(str, instance))) + '\n')

        if 'x_test_vis' in kwargs.keys():
            with open(path.join(base_path, '%s_x_test_vis.txt' % prefix), 'w+') as f:
                for instance in kwargs['x_test_vis']:
                    f.write(' '.join(list(map(str, instance))) + '\n')

        if 'x_test_sem' in kwargs.keys():
            with open(path.join(base_path, '%s_x_test_sem.txt' % prefix), 'w+') as f:
                for instance in kwargs['x_test_sem']:
                    f.write(' '.join(list(map(str, instance))) + '\n')

        if 'y_train' in kwargs.keys():
            with open(path.join(base_path, '%s_y_train.txt' % prefix), 'w+') as f:
                f.write('\n'.join(list(map(str, kwargs['y_train']))))

        if 'y_test' in kwargs.keys():
            with open(path.join(base_path, '%s_y_test.txt' % prefix), 'w+') as f:
                f.write('\n'.join(list(map(str, kwargs['y_test']))))
