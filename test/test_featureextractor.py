'''
Tests for module featureextractor

@author: Damares Resende
@contact: damaresresende@usp.br
@since: May 5, 2019
@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
'''
import os
import unittest
from utils import featureextractor as fe

class ImagesParserTests(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        '''
        Initializes model for all tests
        '''
        self._parser = fe.ImageParser()
        
    def test_preprocess_one_image(self):
        '''
        Tests if image is preprocessed to be inputed into convolutional model
        '''
        input_imgs = [os.getcwd() + '/_mockfiles/awa2/images/fox_10154.jpg']
        stimuli = self._parser.preprocess_batch(input_imgs)
        
        self.assertEqual((1, 224, 224, 3), stimuli.shape)
        
    def test_preprocess_two_images(self):
        '''
        Tests if images are preprocessed to be inputed into convolutional model
        '''
        input_imgs = [os.getcwd() + '/_mockfiles/awa2/images/leopard_10385.jpg', 
                      os.getcwd() + '/_mockfiles/awa2/images/horse_11288.jpg']
        stimuli = self._parser.preprocess_batch(input_imgs)
        
        self.assertEqual((2, 224, 224, 3), stimuli.shape)
        
    def test_classify_image(self):
        '''
        Tests if model is capable of correctly classifying an image
        '''
        input_img = os.getcwd() + '/_mockfiles/awa2/images/fox_10470.jpg'
        labels = self._parser.classify(input_img, labels_only=True)
        self.assertTrue('grey_fox' in labels)
        
    def test_get_activations(self):
        '''
        Tests if hidden layer output can be extracted 
        '''
        layer_name = 'bn5c_branch2c'
        input_img = os.getcwd() + '/_mockfiles/awa2/images/horse_10805.jpg'
        activations = self._parser.get_activations(layer_name, input_img)
        self.assertEqual((1, 7, 7, 2048), activations.shape)
    
    def test_plot_activations(self):
        '''
        Tests if activations are plotted
        '''
        layer_name = 'bn5c_branch2c'
        input_img = os.getcwd() + '/_mockfiles/awa2/images/leopard_10208.jpg'
        file_name = os.getcwd() + '/_mockfiles/awa2/activations_%s.jpg' % layer_name
        
        if os.path.isfile(file_name):
            os.remove(file_name)
        
        activations = self._parser.get_activations(layer_name, input_img)
        self._parser.plot_activations(activations[0,:,:,0:16], file_name)
        self.assertTrue(os.path.isfile(file_name))
        
    def test_get_imageset_features(self):
        '''
        Tests if features from all images in folder are retrieved and save to file
        '''
        layer_name = 'avg_pool'
        input_folder = os.getcwd() + '/_mockfiles/awa2/images/'
        images = [input_folder + img for img in os.listdir(input_folder)]
        activations = self._parser.get_activations(layer_name, images)
        
        self.assertEquals((6, 2048), activations.shape)
        