'''
Retrieves basic information about the Animals With Attributes 2 dataset. The
data retrieved includes all possible classes and attributes.

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Feb 19, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
'''

import numpy as np
import pandas as pd
from enum import Enum
from os.path import join
from utils.logwriter import Logger, MessageType


class PredicateType(Enum):
    '''
    Enum for predicate type
    '''
    BINARY =  1
    CONTINUOUS = 2


class AnnotationsParser():
    
    def __init__(self, base_path):
        '''
        Initialization
        
        @param base_path: string that points to path where the base data files are
        '''
        self.base_path = base_path
        
    def get_labels(self):
        '''
        Retrieves the labels available for objects in Animals with Attributes 2 data set
        
        @return list of strings with available labels
        '''
        try:
            file_path = join(self.base_path, 'classes.txt')
            with open(file_path) as f:
                labels = [line.split()[1] for line in f.readlines()]
                
            return labels
        except FileNotFoundError:
            Logger().write_message('File %s could not be found.' % file_path, MessageType.ERR)
            return []
    
    def get_predicates(self):
        '''
        Retrieves the attributes available for objects in Animals with Attributes 2 data set
        
        @return list of strings with available predicates
        '''
        try:
            file_path = join(self.base_path, 'predicates.txt')
            with open(file_path) as f:
                predicates = [line.split()[1] for line in f.readlines()]
                
            return predicates
        except FileNotFoundError:
            Logger().write_message('File %s could not be found.' % file_path, MessageType.ERR)
            return []
        
    def get_attributes(self, ptype=PredicateType.BINARY):
        '''
        Retrieves data frame with object labels and corresponding attributes
        
        @return pandas data frame with 50 labels and 85 corresponding attributes
        '''
        try:
            if ptype == PredicateType.BINARY:
                file_path = join(self.base_path, 'predicate-matrix-binary.txt')
            elif ptype == PredicateType.CONTINUOUS:
                file_path = join(self.base_path, 'predicate-matrix-continuous.txt')
            else:
                raise ValueError('Invalid predicate type')
            
            with open(file_path) as f:
                matrix = np.zeros((50, 85), dtype=np.float32)
                
                for i, line in enumerate(f.readlines()):
                    for j, value in enumerate(line.split()):
                        matrix[i,j] = float(value)
                
            return pd.DataFrame(data=matrix,
                                index=self.get_labels(),
                                columns=self.get_predicates())
        except FileNotFoundError:
            Logger().write_message('File %s could not be found.' % file_path, MessageType.ERR)
            return None
        
    def get_subset_features(self, ann_dict, ptype=PredicateType.BINARY):
        '''
        Creates a dictionary with a subset of attributes based on the attributes stated in 
        predicates-subset.txt
        
        @param ann_dict: dictionary with the annotations to create a subset of features
        @return pandas data frame with 50 labels and X corresponding attributes
        '''
        features = self.get_attributes(ptype)
        attributes = [att[1] for key in ann_dict.keys() for att in ann_dict[key]]
        subset = pd.DataFrame(index=features.index, columns=attributes)
        
        for att in attributes:
            subset[att] = features[att]
            
        return subset
    
    def get_subset_annotations(self):
        '''
        Creates a dictionary with the annotations to create a subset of features
        
        @return dictionary of size (N, X) per key, where X is the number of features for that key
        and N the number of keys. Each element in the list is a tuple where first value is a numeric
        label and last value the text label
        '''
        try:
            file_path = join(self.base_path, 'predicates-subset.txt')
            attributes = dict()
            key = ''
            
            with open(file_path) as f:
                for line in f.readlines():
                    line = line.strip()
                    
                    if line:
                        if not line[0].isdigit():
                            attributes[line] = []
                            key = line
                        else:
                            labels = line.split()
                            attributes[key].append((int(labels[0]), labels[1]))
                
            return attributes
        except KeyError:
            Logger().write_message('Row %s could not be parsed' % line, MessageType.ERR)
            return None
        except FileNotFoundError:
            Logger().write_message('File %s could not be found.' % file_path, MessageType.ERR)
            return None
