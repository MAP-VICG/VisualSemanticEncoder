"""
Retrieves basic information about the Birds dataset

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Jan 5, 2020

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""
import numpy as np
from os import path


class BirdsData:
    def __init__(self, base_path):
        self.images_path = path.join(base_path, 'images')
        self.lists_path = path.join(base_path, 'lists')
        self.attributes_path = path.join(base_path, 'attributes')
        self.images_list = []

    def get_images_list(self, list_file):
        """
        Reads the file and saves the list of image paths into a list of strings

        @param list_file: string indicating the full path to the file with the list of images
        @return: None
        """
        with open(path.join(self.lists_path, list_file)) as f:
            for line in f.readlines():
                self.images_list.append(line.strip())

    def get_semantic_attributes(self):
        """
        Reads the file of attributes and computes the semantic array for each image taking into account the value of
        the attribute given by each evaluator and the certainty factor indicated. Array is normalized so final result
        lies in between 0 and 1.

        @return: float numpy array of shape (6033, 288)
        """
        features = np.zeros((6033, 288))
        divisors = np.ones((6033, 288))

        with open(path.join(self.attributes_path, 'labels.txt')) as f:
            for line in f.readlines():
                img_id, att_id, att_value, att_certainty, evaluator_id = list(map(int, line.split()))
                features[img_id, att_id] += self.map_evaluation_certainty(att_value, att_certainty)
                divisors[img_id, att_id] += 1

        return features / divisors

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

    def save_class_file(self, file_name):
        """
        From the list of images retrieved it gets the class of each image and saves it into a file

        @param file_name: string indicating the full path to the file to save the data class list
        @return: None
        """
        with open(file_name, 'w+') as f:
            for img_path in self.images_list:
                f.write(str(img_path.split('.')[0]) + '\n')
