"""
Retrieves basic information about the Birds dataset

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Jan 5, 2020

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""
from os import path


class BirdsData:
    def __init__(self, base_path):
        self.images_path = path.join(base_path, 'images')
        self.lists_path = path.join(base_path, 'lists')
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

    def save_class_file(self, file_name):
        """
        From the list of images retrieved it gets the class of each image and saves it into a file

        @param file_name: string indicating the full path to the file to save the data class list
        @return: None
        """
        with open(file_name, 'w+') as f:
            for img_path in self.images_list:
                f.write(str(img_path.split('.')[0]) + '\n')

