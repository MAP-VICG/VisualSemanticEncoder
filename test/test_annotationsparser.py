'''
Tests for module annotationsparser

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Feb 19, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
'''

import unittest
import pandas as pd

from core.annotationsparser import AnnotationsParser, PredicateType


class AnnotationsParserTests(unittest.TestCase):
    
    def test_get_labels(self):
        '''
        Tests if list of labels is retrieved
        '''
        parser = AnnotationsParser('./_mockfiles/awa2/base/')
        labels = parser.get_labels()
        
        self.assertEqual(50, len(labels))
        self.assertEqual('antelope', labels[0])
        self.assertEqual('dolphin', labels[49])
        self.assertEqual('humpback+whale', labels[17])
        
    def test_get_labels_file_not_found(self):
        '''
        Tests if empty list is returned when file is not found
        '''
        parser = AnnotationsParser('./_mockfiles/awa2/base/dummy/')
        labels = parser.get_labels()
        
        self.assertEqual(0, len(labels))
        
    def test_get_predicates(self):
        '''
        Tests if list of attributes is retrieved
        '''
        parser = AnnotationsParser('./_mockfiles/awa2/base/')
        predicates = parser.get_predicates()
        
        self.assertEqual(85, len(predicates))
        self.assertEqual('orange', predicates[5])
        self.assertEqual('tree', predicates[76])
        self.assertEqual('hands', predicates[19])
        
    def test_get_predicates_file_not_found(self):
        '''
        Tests if empty list is returned when file is not found
        '''
        parser = AnnotationsParser('./_mockfiles/awa2/base/dummy/')
        predicates = parser.get_predicates()
        
        self.assertEqual(0, len(predicates))    
    
    def test_get_attributes(self):
        '''
        Tests if data frame with attributes is retrieved
        '''
        parser = AnnotationsParser('./_mockfiles/awa2/base/')
        attributes = parser.get_attributes()
        
        self.assertTrue(isinstance(attributes, pd.DataFrame))
        self.assertEqual(parser.get_labels(), list(attributes.index.values))
        self.assertEqual(parser.get_predicates(), list(attributes.columns.values))
        self.assertEqual((50,), attributes['toughskin'].values.shape)
        self.assertEqual((85,), attributes.loc['gorilla'].values.shape)
        
    def test__attributes_content(self):
        '''
        Tests if data frame with attributes have reasonable values
        '''
        parser = AnnotationsParser('./_mockfiles/awa2/base/')
        attributes = parser.get_attributes(PredicateType.BINARY)
        
        for label in parser.get_labels():
            self.assertTrue(sum(attributes.loc[label].values) < 85)
            for value in attributes.loc[label].values:
                self.assertTrue(value == 0 or value == 1)
            
        for att in parser.get_predicates():
            self.assertTrue(sum(attributes[att].values) < 50)
            for value in attributes.loc[label].values:
                self.assertTrue(value == 0 or value == 1)
