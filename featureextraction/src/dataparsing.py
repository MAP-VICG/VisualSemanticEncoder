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
        self.images_path = path.join(base_path, 'images')
        self.lists_path = path.join(base_path, 'lists')
        self.attributes_path = path.join(base_path, 'attributes')
        self.num_attributes = 288

    @staticmethod
    def get_images_list(list_file):
        """
        Reads the file and saves the list of image paths into a list of strings

        @param list_file: string indicating the full path to the file with the list of images
        @return: list of strings with images' paths
        """
        try:
            with open(list_file) as f:
                images_list = [line.strip().split()[-1] for line in f.readlines() if line]

            return images_list
        except (IOError, FileNotFoundError):
            return None

    def get_semantic_attributes(self, num_images):
        """
        Reads the file of attributes and computes the semantic array for each image taking into account the value of
        the attribute given by each evaluator and the certainty factor indicated. Array is normalized so final result
        lies in between 0 and 1.

        @param num_images: number of images in the data set
        @return: float numpy array of shape (num_images, 288)
        """
        try:
            with open(path.join(self.attributes_path, 'labels.txt')) as f:
                features = np.zeros((num_images, self.num_attributes))
                divisors = np.ones((num_images, self.num_attributes))

                for line in f.readlines():
                    img_id, att_id, att_value, att_certainty, evaluator_id = list(map(int, line.split()))
                    features[img_id, att_id] += self.map_evaluation_certainty(att_value, att_certainty)
                    divisors[img_id, att_id] += 1

            return features / divisors
        except (IOError, FileNotFoundError):
            return None

    @staticmethod
    def map_evaluation_certainty(att_value, att_certainty):
        """
        Computes the true value of an attribute based on the certainty factor. If certainty is 2 (guessing),
        returned value will be 0.5 (random probability). If certainty is 0 (probably), the value will suffer a penalty
        of 0.25. At last, if certainty is 1 (definitely), the value will be itself (0 or 1).

        @param att_value: attribute value (0 or 1)
        @param att_certainty: attribute certainty (0, 1 or 2)
        @return: integer with attribute's true value
        """
        if att_value == 1 and att_certainty == 0:
            return 0.75
        elif att_value == 1 and att_certainty == 1:
            return 1
        elif att_certainty == 2:
            return 0.5
        elif att_value == 0 and att_certainty == 0:
            return 0.25
        return 0

    def get_visual_attributes(self, images_list):
        """
        Extracts the visual features of every image listed using ResNet50 model

        @param images_list: list of strings with image's names
        @return: numpy array of shape (X, 2048) where X is the number of images listed
        """
        vis_data = ResNet50FeatureExtractor(images_list, self.images_path)
        vis_data.extract_images_list_features()
        return vis_data.features_set

    def get_train_test_masks(self, train_list):
        """
        Based on the list of images indicated, goes through the full list of images and creates boolean masks
        to split data set into training and test sets.

        @param train_list: list of strings with image's in training set
        @return: tuple with boolean lists for training and test sets
        """
        full_list = self.get_images_list(path.join(self.attributes_path, 'images-dirs.txt'))

        train_mask = [False] * len(full_list)
        test_mask = [True] * len(full_list)

        for image in train_list:
            idx = full_list.index(image)
            train_mask[idx] = True
            test_mask[idx] = False

        return train_mask, test_mask

    @staticmethod
    def get_images_class(images_list):
        """
        Retrieves the class of each image based on the name of it

        @param images_list: list of images with class id at the beginning
        @return: list of integers
        """
        classes = []
        for img_path in images_list:
            classes.append(int(img_path.split('.')[0].strip()))
        return classes

    def build_birds_data(self, num_images):
        """
        Builds a data set with visual features extracted from ResNet50 and semantic features extracted from labels file

        @param num_images: number of images in the data set including training and test sets
        @return: training data, training class id list, test data, test class id list
        """
        train_list = self.get_images_list(path.join(self.lists_path, 'train.txt'))
        test_list = self.get_images_list(path.join(self.lists_path, 'test.txt'))
        sem_fts = self.get_semantic_attributes(num_images)

        train_mask, test_mask = self.get_train_test_masks(train_list)

        train_vis_fts = self.get_visual_attributes(train_list)
        test_vis_fts = self.get_visual_attributes(test_list)
        train_sem_fts = sem_fts[train_mask]
        test_sem_fts = sem_fts[test_mask]

        return np.hstack((train_vis_fts, train_sem_fts)), self.get_images_class(train_list), \
               np.hstack((test_vis_fts, test_sem_fts)), self.get_images_class(test_list)

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
