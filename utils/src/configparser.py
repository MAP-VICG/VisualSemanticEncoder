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

from encoding.src.encoder import ModelType


class ConfigParser:
    def __init__(self, configfile):
        """
        initializes parameters to default values
        
        @param configfile: path to the configuration XML file
        """
        if not configfile.endswith('.xml'):
            raise ValueError('Configuration file must be in XML format.')
        
        self.epochs = 0
        self.console = False
        self.encoding_size = 0
        self.output_size = 2048
        self.ae_type = None

        self.dataset = ''
        self.results_path = ''
        self.x_train_path = ''
        self.y_train_path = ''
        self.x_test_path = ''
        self.y_test_path = ''

        self.classes_names = None
        self.chosen_classes = None
        self.configfile = configfile
        self.baseline = {'vis': 0.0, 'stk': 0.0, 'tnn': 0.0, 'pca': 0.0}
        
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

    def set_dataset_name(self, root):
        """
        Reads XML looking for dataset node and sets its value

        @param root: XML root node
        @return None
        """
        try:
            self.dataset = root.find('general/dataset').text
        except AttributeError:
            raise AttributeError('Could not find "dataset" node')

    def set_autoencoder_type(self, root):
        """
        Reads XML looking for ae_type node and sets its value.

        @param root: XML root node
        @return None
        """
        try:
            ae_type = root.find('autoencoder/ae_type').text
            if ae_type == ModelType.EXTENDED_AE.value:
                self.ae_type = ModelType.EXTENDED_AE
            elif ae_type == ModelType.SIMPLE_AE.value:
                self.ae_type = ModelType.SIMPLE_AE
            elif ae_type == ModelType.SIMPLE_VAE.value:
                self.ae_type = ModelType.SIMPLE_VAE
            elif ae_type == ModelType.EXTENDED_VAE.value:
                self.ae_type = ModelType.EXTENDED_VAE
            else:
                raise ValueError('Invalid type of model chosen.')
        except AttributeError:
            raise AttributeError('Could not find "ae_type" node')

    def set_num_epochs(self, root):
        """
        Reads XML looking for epochs node and sets its value

        @param root: XML root node
        @return None
        """
        try:
            self.epochs = int(root.find('autoencoder/epochs').text)
        except AttributeError:
            raise AttributeError('Could not find "epochs" node')

    def set_chosen_classes(self, root):
        """
        Reads XML looking for chosen_classes node and sets its value
        
        @param root: XML root node
        @return None
        """
        try:
            if not self.dataset and not isinstance(self.dataset, str):
                raise ValueError('Data set tag must be set under general node')

            path = 'semantic_features/%s/chosen_classes' % self.dataset
            self.chosen_classes = list(map(int, root.find(path).text.split(',')))
        except AttributeError:
            raise AttributeError('Could not find "chosen_classes" node')

    def set_classes_names(self, root):
        """
        Reads XML looking for classes_names node and sets its value

        @param root: XML root node
        @return None
        """
        try:
            if not self.dataset and not isinstance(self.dataset, str):
                raise ValueError('Data set tag must be set under general node')

            path = 'semantic_features/%s/classes_names' % self.dataset
            self.classes_names = list(map(str, root.find(path).text.split(',')))
            self.classes_names = [name.strip() for name in self.classes_names]
        except AttributeError:
            raise AttributeError('Could not find "classes_names" node')

    def set_baselines(self, root):
        """
        Reads XML looking for baselines node and sets its value

        @param root: XML root node
        @return None
        """
        try:
            if not self.dataset and not isinstance(self.dataset, str):
                raise ValueError('Data set tag must be set under general node')

            if not self.encoding_size and not isinstance(self.encoding_size, int):
                raise ValueError('Encoding size tag must be set under autoencoder node')

            path = 'baselines/%s/' % self.dataset
            self.baseline['vis'] = float(root.find(path + 'vis').text)
            self.baseline['stk'] = float(root.find(path + 'stk').text)
            self.baseline['tnn'] = float(root.find(path + 'tnn').text)
            self.baseline['pca'] = float(root.find(path + 'pca/c' + str(self.encoding_size)).text)
        except AttributeError:
            raise AttributeError('Could not find "baselines" node')
        
    def set_encoding_size(self, root):
        """
        Reads XML looking for encoding_size node and sets its value
        
        @param root: XML root node
        @return None
        """
        try:
            self.encoding_size = int(root.find('autoencoder/encoding_size').text)
        except AttributeError:
            raise AttributeError('Could not find "encoding_size" node')

    def set_output_size(self, root):
        """
        Reads XML looking for output_size node and sets its value

        @param root: XML root node
        @return None
        """
        try:
            self.output_size = int(root.find('autoencoder/output_size').text)
        except AttributeError:
            raise AttributeError('Could not find "output_size" node')
        
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

    def set_x_train_path(self, root):
        """
        Reads XML looking for x_train_path node and sets its value

        @param root: XML root node
        @return None
        """
        try:
            self.x_train_path = root.find('paths/x_train_path').text
            self.x_train_path.replace('/', os.sep)
            self.x_train_path.replace('\\', os.sep)
        except AttributeError:
            raise AttributeError('Could not find "x_train_path" node')

    def set_y_train_path(self, root):
        """
        Reads XML looking for y_train_path node and sets its value

        @param root: XML root node
        @return None
        """
        try:
            self.y_train_path = root.find('paths/y_train_path').text
            self.y_train_path.replace('/', os.sep)
            self.y_train_path.replace('\\', os.sep)
        except AttributeError:
            raise AttributeError('Could not find "y_train_path" node')

    def set_x_test_path(self, root):
        """
        Reads XML looking for x_test_path node and sets its value

        @param root: XML root node
        @return None
        """
        try:
            self.x_test_path = root.find('paths/x_test_path').text
            self.x_test_path.replace('/', os.sep)
            self.x_test_path.replace('\\', os.sep)
        except AttributeError:
            raise AttributeError('Could not find "x_test_path" node')

    def set_y_test_path(self, root):
        """
        Reads XML looking for y_test_path node and sets its value

        @param root: XML root node
        @return None
        """
        try:
            self.y_test_path = root.find('paths/y_test_path').text
            self.y_test_path.replace('/', os.sep)
            self.y_test_path.replace('\\', os.sep)
        except AttributeError:
            raise AttributeError('Could not find "y_test_path" node')

    def read_configuration(self):
        """
        Reads XML looking for the configuration nodes to set their respective values
        
        @return None
        """
        tree = ET.parse(self.configfile)
        root = tree.getroot()

        self.set_dataset_name(root)
        self.set_console_value(root)
        self.set_autoencoder_type(root)

        self.set_num_epochs(root)
        self.set_encoding_size(root)
        self.set_output_size(root)

        self.set_results_path(root)
        self.set_x_train_path(root)
        self.set_y_train_path(root)
        self.set_x_test_path(root)
        self.set_y_test_path(root)

        if self.dataset and self.encoding_size:
            self.set_baselines(root)
            self.set_chosen_classes(root)
            self.set_classes_names(root)
