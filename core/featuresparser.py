'''
Retrieves features of 37322 images extracted with ResNet101. Each feature vector has
2048 features.

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Feb 19, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
'''

import numpy as np
from os.path import join
from keras.utils import normalize

from core.annotationsparser import AnnotationsParser, PredicateType


class FeaturesParser():
    
    def __init__(self, features_path):
        '''
        Initialization
        
        @param base_path: string that points to path where the features data files are
        '''
        self.features_path = features_path
        
    def get_labels(self):
        '''
        Retrieves the labels of each image in Animals with Attributes 2 data set
        
        @return numpy array of integers with labels
        '''
        try:
            file_path = join(self.features_path, 'AwA2-labels.txt')
            with open(file_path) as f:
                lines = f.readlines()
                labels = np.zeros((len(lines),), dtype=np.int32)
                
                for idx, line in enumerate(lines):
                    labels[idx] = int(line)
                
            return labels
        except FileNotFoundError:
            print('>> ERROR: file %s could not be found.' % file_path)
            return None
        
    def get_visual_features(self, norm=False, norm_axis=1):
        '''
        Retrieves features extracted by ResNet101
        
        @param norm: normalize features
        @return numpy array of shape (37322, 2048) with features for images in AwA2 data set
        '''
        try:
            file_path = join(self.features_path, 'AwA2-features.txt')
            with open(file_path) as f:
                lines = f.readlines()
                features = np.zeros((len(lines), 2048), dtype=np.float32)
                
                for i, line in enumerate(lines):
                    for j, value in enumerate(line.split()):
                        features[i, j] = float(value)
                
            if norm:
                print('>> Normalizing visual features')
                return normalize(features, order=2, axis=norm_axis)
        
            return features
        except FileNotFoundError:
            print('>> ERROR: file %s could not be found.' % file_path)
            return None
    
    def get_semantic_features(self, ann_path, ptype=PredicateType.BINARY, norm=False, norm_axis=1):
        '''
        Retrieves semantic features based on annotations
        
        @param ann_path: annotation path
        @param ptype: predicate type (binary or continuous)
        @param norm: normalize features
        @return numpy array of shape (37322, 85) with features for images in AwA2 data set
        '''
        ann_parser = AnnotationsParser(ann_path)
        att_map = ann_parser.get_attributes(ptype)
        labels = ann_parser.get_labels()
        available_labels = self.get_labels()
        
        features = np.zeros((available_labels.shape[0], 85), dtype=np.float32)
        for idx, label in enumerate(available_labels):
            features[idx, :] = att_map.loc[labels[label-1]].values
        
        if norm:
            print('>> Normalizing semantic features')
            return normalize(features, order=2, axis=norm_axis)
        return features
    
    def concatenate_features(self, vis_fts, sem_fts):
        '''
        Concatenates semantic and visual features along x axis
        
        @param vis_fts: visual features
        @param sem_fts: semantic features
        @return: numpy array of shape (37322, 2048 + 85) with all features
        '''
        features = np.zeros((vis_fts.shape[0], 2048 + 85), dtype=np.float32)
         
        for ft in range(features.shape[0]):
            features[ft,:vis_fts.shape[1]] = vis_fts[ft]
            features[ft,vis_fts.shape[1]:] = sem_fts[ft] 
  
        return features