"""
Retrieves visual features extracted from an image via ResNet50 or InceptionV3 neural networks built in Keras

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Jan 5, 2020

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""
import numpy as np
from os import path
from enum import Enum
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import resnet50
from tensorflow.keras.applications import inception_v3


class ExtractionType(Enum):
    RESNET = "resnet"
    INCEPTION = "inception"

    def __str__(self):
        return self.value

    @staticmethod
    def from_string(s):
        try:
            return ExtractionType[s]
        except KeyError:
            raise ValueError()


class ExtractorFactory:
    def __init__(self, base_path='.'):
        self.base_path = base_path

    def __call__(self, extractor):
        if extractor == ExtractionType.RESNET:
            return ResNet50FeatureExtractor(self.base_path)
        if extractor == ExtractionType.INCEPTION:
            return InceptionV3FeatureExtractor(self.base_path)


class FeatureExtractor:
    def __init__(self, base_path='.'):
        """
        Initializes main variables

        @param base_path: string pointing to the path where the images are located. If no string is indicated, the
            current directory will be considered
        """
        self.base_path = base_path

    def extract_image_features(self, img_path):
        """
        Loads the indicated image and extracts its visual features with ResNet50 model

        @param img_path: string with path to image
        @return: numpy array with image's extracted features
        """
        pass

    def extract_images_list_features(self, images_list):
        """
        Extracts the visual features of all images indicated in images_list and saves the array in features_set

        @param images_list: list of string with names of images to extract features from
        @return: 2D numpy array with extracted features
        """
        features_set = self.extract_image_features(path.join(self.base_path, images_list[0]))
        for img_path in images_list[1:]:
            features = self.extract_image_features(path.join(self.base_path, img_path))
            features_set = np.vstack((features_set, features))

        return features_set


class ResNet50FeatureExtractor(FeatureExtractor):
    def __init__(self, base_path='.'):
        """
        Initializes main variables

        @param base_path: string pointing to the path where the images are located. If no string is indicated, the
            current directory will be considered
        """
        super(ResNet50FeatureExtractor, self).__init__(base_path)
        self.model = resnet50.ResNet50(weights='imagenet', include_top=False, pooling='avg')

    def extract_image_features(self, img_path):
        """
        Loads the indicated image and extracts its visual features with ResNet50 model

        @param img_path: string with path to image
        @return: numpy array with image's extracted features
        """
        img = image.load_img(img_path, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = resnet50.preprocess_input(img_data)
        return self.model.predict(img_data)


class InceptionV3FeatureExtractor(FeatureExtractor):
    def __init__(self, base_path='.'):
        """
        Initializes main variables

        @param base_path: string pointing to the path where the images are located. If no string is indicated, the
            current directory will be considered
        """
        super(InceptionV3FeatureExtractor, self).__init__(base_path)
        self.model = inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling='avg')

    def extract_image_features(self, img_path):
        """
        Loads the indicated image and extracts its visual features with ResNet50 model

        @param img_path: string with path to image
        @return: numpy array with image's extracted features
        """
        img = image.load_img(img_path, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = inception_v3.preprocess_input(img_data)
        return self.model.predict(img_data)
