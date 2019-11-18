'''
Reads attributes in config.xml file to set configuration parameters

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Nov 17, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
'''
from enum import Enum
import xml.etree.ElementTree as ET


class AttributesType(Enum):
    '''
    Enum for attributes type
    '''
    CON = 'continuous'
    BIN = 'binary'
    IND = 'indexed'

    
class ConfigParser:
    def __init__(self, configfile):
        '''
        initializes parameters to default values
        
        @param configfile: path to the configuration XML file
        '''
        if not configfile.endswith('.xml'):
            raise ValueError('Configuration file must be in XML format.')
        
        self.configfile = configfile
        self.mock = False
        self.epochs = 0
        self.encoding_size = 0
        self.noise_rate = 0
        self.attributes_type = ''
        
    def set_mock_value(self, root):
        '''
        Reads XML looking for mock node and sets its value
        
        @param root: XML root node
        @return None
        '''
        try:
            self.mock = bool(root.find('config/mock').text)
        except AttributeError:
            raise AttributeError('Could not find "mock" node')
        
    def set_num_epocks(self, root):
        '''
        Reads XML looking for epochs node and sets its value
        
        @param root: XML root node
        @return None
        '''    
        try:
            self.epochs = int(root.find('config/epochs').text)
        except AttributeError:
            raise AttributeError('Could not find "epochs" node')
        
    def set_encoding_size(self, root):
        '''
        Reads XML looking for encoding_size node and sets its value
        
        @param root: XML root node
        @return None
        '''    
        try:
            self.encoding_size = int(root.find('config/encoding_size').text)
        except AttributeError:
            raise AttributeError('Could not find "encoding_size" node')
        
    def set_noise_rate(self, root): 
        '''
        Reads XML looking for noise_rate node and sets its value
        
        @param root: XML root node
        @return None
        '''      
        try:
            self.noise_rate = float(root.find('config/noise_rate').text)
            if not 0 <= self.noise_rate <= 1:
                raise ValueError('Invalid value for noise rate. Rate should be between 0 and 1')
        except AttributeError:
            raise AttributeError('Could not find "noise_rate" node')
        
    def set_attributes_type(self, root):
        '''
        Reads XML looking for attributes_type node and sets its value
        
        @param root: XML root node
        @return None
        '''      
        try:
            att_type = root.find('config/attributes_type').text
        except AttributeError:
            raise AttributeError('Could not find "attributes_type" node')
        
        if att_type == AttributesType.CON.value:
            self.attributes_type = AttributesType.CON
        elif att_type == AttributesType.BIN.value:
            self.attributes_type = AttributesType.BIN
        elif att_type == AttributesType.IND.value:
            self.attributes_type = AttributesType.IND
        else:
            att_types = ', '.join([atype.value for atype in list(AttributesType.__members__.values())])
            raise ValueError('Invalid attributes type. Available options are %s' % att_types)
        
    def read_configuration(self):
        '''
        Reads XML looking for the configuration nodes to set their respective values
        
        @return None
        '''
        tree = ET.parse(self.configfile)
        root = tree.getroot()
        
        self.set_mock_value(root)
        self.set_num_epocks(root)
        self.set_encoding_size(root)
        self.set_noise_rate(root)
        self.set_attributes_type(root)
    
            