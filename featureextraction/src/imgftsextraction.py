"""
Retrieves visual features extracted from an image via ResNet50 neural network built in Keras

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Jan 5, 2020

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""
import numpy as np
from os import path
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input


class ResNet50FeatureExtractor:
    def __init__(self, images_list, base_path='.'):
        """
        Initializes main variables

        @param images_list: list of string with names of images to extract features from
        @param base_path: string pointing to the path where the images are located. If no string is indicated, the
            current directory will be considered
        """
        self.model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        self.images_list = images_list
        self.base_path = base_path
        self.features_set = None

    def extract_image_features(self, img_path):
        """
        Loads the indicated image and extracts its visual features with ResNet50 model

        @param img_path: string with path to image
        @return: numpy array with image's extracted features
        """
        img = image.load_img(img_path, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        return self.model.predict(img_data)

    def extract_images_list_features(self):
        """
        Extracts the visual features of all images indicated in images_list and saves the array in features_set

        @return: None
        """
        self.features_set = self.extract_image_features(path.join(self.base_path, self.images_list[0]))
        for img_path in self.images_list[1:]:
            features = self.extract_image_features(path.join(self.base_path, img_path))
            self.features_set = np.vstack((self.features_set, features))

    def save_features(self, file_path):
        """
        Saves the features_set structure into the indicated file path where each row represents the list of features
        extracted from an image. The values are separated by spaces.

        @param file_path: string indicating the path to the file where the extracted features must be saved
        @return: None
        """
        with open(file_path, 'w+') as f:
            for img_feature in self.features_set:
                f.write(' '.join(list(map(str, img_feature))) + '\n')
