"""
Retrieves basic information about the Birds data set

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Jan 5, 2020

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""
import numpy as np
from os import path

from featureextraction.src.kerasextraction import ResNet50FeatureExtractor


class BirdsData:
    def __init__(self, base_path):
        self.base_path = base_path
        self.images_path = path.join(base_path, 'images')
        self.attributes_path = path.join(base_path, 'attributes')

    def get_semantic_attributes(self):
        """
        Reads the file of attributes and retrieves the semantic array for each class

        @return: float numpy array of shape (200, 312) being 200 the number of classes and 312 the number of attributes
        """
        try:
            with open(path.join(self.attributes_path, 'class_attribute_labels_continuous.txt')) as f:
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
        vis_data = ResNet50FeatureExtractor(images_list, self.images_path)
        vis_data.extract_images_list_features()
        return vis_data.features_set

    def get_train_test_mask(self):
        """
        Goes through the full list of images and creates a boolean mask to split data set into training and test sets,
        where True indicates training set and False test set.

        @return: integer 1D numpy array
        """

        with open(path.join(self.base_path, 'train_test_split.txt')) as f:
            mask = [int(value.split()[1]) for value in f.readlines()]

        return np.array(mask) == 1

    def get_images_class(self):
        """
        Retrieves the class of each image

        @return: integer 1D numpy array
        """
        with open(path.join(self.base_path, 'image_class_labels.txt')) as f:
            klass = [int(value.split()[1]) for value in f.readlines()]

        return np.array(klass)

    def get_images_list(self):
        """
        Retrieves path of each image

        @return: integer 1D numpy array
        """
        with open(path.join(self.base_path, 'images.txt')) as f:
            images_list = [value.split()[1] for value in f.readlines()]

        return np.array(images_list)

    def build_birds_data(self):
        """
        Builds a data set with visual features extracted from ResNet50 and semantic features extracted from labels file

        @return: training data, training class id list, test data, test class id list
        """
        mask = self.get_train_test_mask()
        images = self.get_images_list()
        classes = self.get_images_class()

        sem_fts = self.get_semantic_attributes()
        vis_fts = self.get_visual_attributes(images)

        all_fts = []
        for i, fts in enumerate(vis_fts):
            klass = classes[i] - 1
            all_fts.append(np.hstack((fts, sem_fts[klass, :])))

        all_fts = np.array(all_fts)
        inverse_mask = mask == False

        return all_fts[mask], classes[mask], all_fts[inverse_mask], classes[inverse_mask]

    @staticmethod
    def save_files(base_path, x_train, y_train, x_test, y_test):
        """
        Saves sets data into files

        @param base_path: string with base path to save files
        @param x_train: numpy array with training set
        @param y_train: list of classes
        @param x_test: numpy array with test set
        @param y_test: list of classes
        @return: None
        """
        with open(path.join(base_path, 'birds_x_train.txt'), 'w+') as f:
            for instance in x_train:
                f.write(' '.join(list(map(str, instance))) + '\n')

        with open(path.join(base_path, 'birds_x_test.txt'), 'w+') as f:
            for instance in x_test:
                f.write(' '.join(list(map(str, instance))) + '\n')

        with open(path.join(base_path, 'birds_y_train.txt'), 'w+') as f:
            f.write('\n'.join(list(map(str, y_train))))

        with open(path.join(base_path, 'birds_y_test.txt'), 'w+') as f:
            f.write('\n'.join(list(map(str, y_test))))


class DataParser:
    @staticmethod
    def get_features(data_file):
        """
        Builds a 2D numpy array with data found in file

        @param data_file: string with full path to data file
        @return: float numpy array of shape (X, Y) where X is the number of images and Y the number of attributes
        """
        data = []
        with open(data_file) as f:
            for line in f.readlines():
                data.append(list(map(float, line.split())))
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
