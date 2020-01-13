"""
Retrieves basic information about the Animals With Attributes 2 dataset. The
data retrieved includes all possible classes and attributes.

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Feb 19, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""
import os
import numpy as np
import pandas as pd

from utils.src.logwriter import LogWritter, MessageType


class AnnotationsParser:
    def __init__(self, console=False, binary=False):
        """
        Initialization
        
        @param console: if True, prints debug in console
        @param binary: if True loads the binary predicate matrix, loads the continuous one otherwise 
        """
        base_path = os.path.join(os.path.join(os.path.join(os.getcwd().split('SemanticEncoder')[0],
                                                           'SemanticEncoder'), '_files'), 'base')
        
        self.logger = LogWritter(console=console)
        self.classes = os.path.join(base_path, 'AwA2-classes.txt')
        self.predicate_set = os.path.join(base_path, 'AwA2-predicate-set.txt')
        self.predicate_subset = os.path.join(base_path, 'AwA2-predicate-subset.txt')
        
        if binary:
            self.predicate_matrix = os.path.join(base_path, 'AwA2-predicate-matrix-binary.txt')
        else:
            self.predicate_matrix = os.path.join(base_path, 'AwA2-predicate-matrix-continuous.txt')
        
    def get_classes(self):
        """
        Retrieves the labels available for objects in Animals with Attributes 2 data set
        
        @return list of strings with available labels
        """
        try:
            with open(self.classes) as f:
                labels = [line.strip().split()[1] for line in f.readlines()]
                
            return labels
        except FileNotFoundError:
            self.logger.write_message('File %s could not be found.' % self.classes, MessageType.ERR)
            return []
        
    def get_predicate_matrix(self, subset=False):
        """
        Retrieves data frame with object labels and corresponding attributes
        
        @return pandas data frame with 50 labels and 85 corresponding attributes
        """
        try:
            with open(self.predicate_matrix) as f:
                matrix = np.zeros((50, 85), dtype=np.float32)
                
                for i, line in enumerate(f.readlines()):
                    for j, value in enumerate(line.strip().split()):
                        matrix[i,j] = float(value)
                
            predicates = pd.DataFrame(data=matrix, index=self.get_classes(), columns=self.get_attributes_set())
            
            if subset:
                subset = pd.DataFrame(index=predicates.index, columns=self.get_attributes_subset())
                 
                for att in subset.columns:
                    subset[att] = predicates[att]
                return subset
            
            return predicates
        except FileNotFoundError:
            self.logger.write_message('File %s could not be found.' % self.predicate_matrix, MessageType.ERR)
            return None
        
    def get_attributes_set(self):
        """
        Retrieves the attributes available for objects in Animals with Attributes 2 data set
        
        @return list of strings with available predicates
        """
        try:
            with open(self.predicate_set) as f:
                predicates = [line.strip().split()[1] for line in f.readlines()]
                
            return predicates
        except FileNotFoundError:
            self.logger.write_message('File %s could not be found.' % self.predicate_subset, MessageType.ERR)
            return []
        
    def get_attributes_subset(self, as_dict=False):
        """
        Retrieves the attributes to be considered for objects in Animals with Attributes 2 data set
        
        @param as_dict: if True, returns a dictionary with features category as key
        @return list of strings with predicates to be considered
        """
        try:
            with open(self.predicate_subset) as f:
                if as_dict:
                    key = ''
                    predicates = dict()
                    
                    for line in f.readlines():
                        line = line.strip()
                         
                        if line:
                            if not line[0].isdigit():
                                predicates[line] = []
                                key = line
                            else:
                                labels = line.split()
                                predicates[key].append((int(labels[0]), labels[1]))
                else:
                    predicates = []
                    
                    for line in f.readlines():
                        line = line.strip()
                        if line and line[0].isdigit():
                            value = line.split()[1]
                            if value not in predicates:
                                predicates.append(value)
                
            return predicates
        except KeyError:
            self.logger.write_message('Row %s could not be parsed' % line, MessageType.ERR)
            return None
        except FileNotFoundError:
            self.logger.write_message('File %s could not be found.' % self.predicate_subset, MessageType.ERR)
            return []
