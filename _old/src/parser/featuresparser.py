"""
Retrieves features of 37322 images extracted with ResNet101. Each feature vector has
2048 features.

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Feb 19, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""
import os
import numpy as np

from utils.src.logwriter import LogWritter, MessageType
from _old.src.parser import AnnotationsParser


class FeaturesParser:
    def __init__(self, fts_dir=None, console=False):
        """
        Initialization
        
        @param fts_dir: string that points to folder where the features data files are
        @param console: if True, prints debug in console
        """
        if fts_dir and isinstance(fts_dir, str):
            base_path = os.path.join(os.path.join(os.getcwd().split('SemanticEncoder')[0], 'SemanticEncoder'), fts_dir)
        else:
            base_path = os.path.join(os.path.join(os.getcwd().split('SemanticEncoder')[0], 'SemanticEncoder'), 'features')
        
        self.features = os.path.join(base_path, 'AwA2-features.txt')
        self.filenames = os.path.join(base_path, 'AwA2-filenames.txt')
        self.labels = os.path.join(base_path, 'AwA2-labels.txt')
        self.logger = LogWritter(console=console)
        self.console = console
        
    def get_labels(self):
        """
        Retrieves the labels of each image in Animals with Attributes 2 data set
        
        @return numpy array of integers with labels
        """
        try:
            with open(self.labels) as f:
                lines = f.readlines()
                labels = np.zeros((len(lines),), dtype=np.int32)
                
                for idx, line in enumerate(lines):
                    labels[idx] = int(line)
                
            return labels
        except FileNotFoundError:
            self.logger.write_message('File %s could not be found.' % self.labels, MessageType.ERR)
            return None
        
    def get_visual_features(self):
        """
        Retrieves features extracted by ResNet101
        
        @param norm: normalize features
        @return numpy array of shape (37322, 2048) with features for images in AwA2 data set
        """
        try:
            with open(self.features) as f:
                lines = f.readlines()
                features = np.zeros((len(lines), 2048), dtype=np.float32)
                
                for i, line in enumerate(lines):
                    for j, value in enumerate(line.split()):
                        features[i, j] = float(value)

            return features
        except FileNotFoundError:
            self.logger.write_message('File %s could not be found.' % self.features, MessageType.ERR)
            return None
    
    def get_semantic_features(self, subset=False, binary=False):
        """
        Retrieves semantic features based on annotations
        
        @param subset: if True return a subset of the features with 19 attributes only
        @param binary: if True loads the binary predicate matrix, loads the continuous one otherwise
        @return numpy array of shape (37322, X) with features for images in AwA2 data set where X
                is the number of attributes considered
        """
        ann_parser = AnnotationsParser(self.console, binary=binary)
        
        if subset:
            self.logger.write_message('Using a subset of the semantic features', MessageType.INF)
            att_map = ann_parser.get_predicate_matrix(subset=True)
        else:
            self.logger.write_message('Using whole set of the semantic features', MessageType.INF)
            att_map = ann_parser.get_predicate_matrix()
        
        classes = ann_parser.get_classes()
        labels = self.get_labels()
        features = np.zeros((labels.shape[0], att_map.shape[1]), dtype=np.float32)
        
        for idx, label in enumerate(labels):
            features[idx, :] = att_map.loc[classes[label-1]].values
            
        return features

    @staticmethod
    def concatenate_features(vis_fts, sem_fts):
        """
        Concatenates semantic and visual features along x axis
        
        @param vis_fts: visual features
        @param sem_fts: semantic features
        @return: numpy array of shape (37322, 2048 + X) with all features, where X is the number of
            semantic features
        """
        features = np.zeros((vis_fts.shape[0], vis_fts.shape[1] + sem_fts.shape[1]), dtype=np.float32)
         
        for ft in range(features.shape[0]):
            features[ft, :vis_fts.shape[1]] = vis_fts[ft]
            features[ft, vis_fts.shape[1]:] = sem_fts[ft]
  
        return features
