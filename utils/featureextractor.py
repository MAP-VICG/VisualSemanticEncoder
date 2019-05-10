'''
Extracts features from the specified layer from the ResNet50 model

@author: Damares Resende
@contact: damaresresende@usp.br
@since: May 5, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.models import Model
from keras.preprocessing import image
from keras.layers import Input, GlobalAveragePooling2D
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions


class ImageParser():
    
    def __init__(self):
        '''
        Initializes ResNet50 model trained in imagenet dataset
        '''
        self.width = 224
        self.height = 224
        self.model = None
        
        
    def load_resnet(self):
        '''
        Loads ResNet50 model
        '''
        if not self.model:
            self.model = ResNet50(weights='imagenet')

    def preprocess_image(self, input_img):
        '''
        Loads a image and sets it to the shape of ResNet input
        
        @param input_img: string with image complete path
        @return image in shape (1, 224, 224, x) where x is the number of channels in the image
        '''
        img = image.load_img(input_img, target_size=(self.width, self.height))
        img = image.img_to_array(img)
            
        return preprocess_input(np.expand_dims(img, axis=0))
    
    def preprocess_batch(self, input_imgs):
        '''
        Loads a list of RGB images and creates a batch with them in the shape
        of ResNet input
        
        @param input_imgs: list of strings with the complete path of images to load
        @return batch of images in shape (n, 224, 224, 3) where n is the number of
        input images  
        '''
        images = np.zeros((0, self.width, self.height, 3))
        
        for img_file in input_imgs:
            images = np.vstack([images, self.preprocess_image(img_file)])
            
        return images
        
    def classify(self, input_img, labels_only=False):
        '''
        Classifies an image with ResNet50 model trained on imagenet dataset
        
        @param input_img: string or list of strings with images complete path
        @param labels_only: if true returns only labels, otherwise returns labels and probabilities
        @result list of possible labels of the input image if labels_only is True or list 
        of labels and probabilities
        '''
        self.load_resnet()
        if isinstance(input_img, list):
            batch = self.preprocess_batch(input_img)
        else:
            batch = self.preprocess_image(input_img)
            
        batch = preprocess_input(batch)
        labels = decode_predictions(self.model.predict(batch), top=3)[0]

        if labels_only:
            return [label[1] for label in labels]
        else:
            return labels
        
    def get_activations(self, layer_name, input_img):
        '''
        Gets the result of activations in a specific layer
        
        @param layer_name: string with layer name
        @param input_img: string or list of strings with image complete path
        @return output of activation filters of specified layer 
        '''
        self.load_resnet()
        if isinstance(input_img, list):
            stimuli = self.preprocess_batch(input_img)
        else:
            stimuli = self.preprocess_image(input_img)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            
            output = self.model.get_layer(layer_name).output
            filters = sess.run(output, feed_dict={'input_1:0': stimuli})
            
        return filters
    
    def plot_activations(self, activations, file_name):
        '''
        Plots outputs of 16 filters from the activations specified
        
        @param activations: layer output
        @param file_name: output file name 
        '''
        plt.figure(1, figsize=(20, 20))
        for i in range(16):
            plt.subplot(4, 4, i+1)
            plt.title('Filter ' + str(i))
            plt.gray()
            plt.imshow(activations[:,:,i])
        plt.savefig(file_name)
        
    def __del__(self):
        '''
        Cleans tensorflow graph
        '''
        tf.reset_default_graph()