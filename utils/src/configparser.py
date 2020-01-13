"""
Reads attributes in config.xml file to set configuration parameters

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Nov 17, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""
import os
import xml.etree.ElementTree as ET

    
class ConfigParser:
    def __init__(self, configfile):
        """
        initializes parameters to default values
        
        @param configfile: path to the configuration XML file
        """
        if not configfile.endswith('.xml'):
            raise ValueError('Configuration file must be in XML format.')
        
        self.epochs = 0
        self.noise_rate = 0
        self.console = False
        self.results_path = ''
        self.encoding_size = 0
        self.features_path = ''
        self.save_test_set = True
        self.attributes_type = None
        self.configfile = configfile
        self.ae_noise_factor = 0
        
    def set_console_value(self, root):
        """
        Reads XML looking for console node and sets its value
        
        @param root: XML root node
        @return None
        """
        try:
            flag = root.find('general/console').text
            if flag == 'False':
                self.console = False
            elif flag == 'True':
                self.console = True
            else:
                raise ValueError('Invalid value for console. Please choose True or False')
        except AttributeError:
            raise AttributeError('Could not find "console" node')
        
    def set_num_epochs(self, root):
        """
        Reads XML looking for epochs node and sets its value
        
        @param root: XML root node
        @return None
        """
        try:
            self.epochs = int(root.find('auto_encoder/epochs').text)
        except AttributeError:
            raise AttributeError('Could not find "epochs" node')

    def set_noise_factor(self, root):
        """
        Reads XML looking for AE noise factor node and sets its value

        @param root: XML root node
        @return None
        """
        try:
            self.ae_noise_factor = float(root.find('auto_encoder/noise_factor').text)
        except AttributeError:
            raise AttributeError('Could not find "noise_factor" node')
        
    def set_encoding_size(self, root):
        """
        Reads XML looking for encoding_size node and sets its value
        
        @param root: XML root node
        @return None
        """
        try:
            self.encoding_size = int(root.find('auto_encoder/encoding_size').text)
        except AttributeError:
            raise AttributeError('Could not find "encoding_size" node')
        
    def set_noise_rate(self, root): 
        """
        Reads XML looking for noise_rate node and sets its value
        
        @param root: XML root node
        @return None
        """
        try:
            self.noise_rate = float(root.find('semantic_features/noise_rate').text)
            if not 0 <= self.noise_rate <= 1:
                raise ValueError('Invalid value for noise rate. Rate should be between 0 and 1')
        except AttributeError:
            raise AttributeError('Could not find "noise_rate" node')
        
    def set_results_path(self, root): 
        """
        Reads XML looking for results_path node and sets its value
        
        @param root: XML root node
        @return None
        """
        try:
            self.results_path = root.find('paths/results_path').text
            self.results_path.replace('/', os.sep)
            self.results_path.replace('\\', os.sep)
        except AttributeError:
            raise AttributeError('Could not find "results_path" node')
        
    def set_features_path(self, root): 
        """
        Reads XML looking for features_path node and sets its value
        
        @param root: XML root node
        @return None
        """
        try:
            self.features_path = root.find('paths/features_path').text
            self.features_path.replace('/', os.sep)
            self.features_path.replace('\\', os.sep)
        except AttributeError:
            raise AttributeError('Could not find "features_path" node')
        
    def read_configuration(self):
        """
        Reads XML looking for the configuration nodes to set their respective values
        
        @return None
        """
        tree = ET.parse(self.configfile)
        root = tree.getroot()

        self.set_num_epochs(root)
        self.set_noise_rate(root)
        self.set_noise_factor(root)
        self.set_results_path(root)
        self.set_console_value(root)
        self.set_encoding_size(root)
        self.set_features_path(root)