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

from core.annotationsparser import AnnotationsParser


class AnnotationsParserTests(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        '''
        Initializes model for all tests
        '''
        cls.parser = AnnotationsParser(console=True)
    
    def test_get_labels(self):
        '''
        Tests if list of labels is retrieved
        '''
        labels = self.parser.get_labels()
        
        self.assertEqual(50, len(labels))
        self.assertEqual('antelope', labels[0])
        self.assertEqual('dolphin', labels[49])
        self.assertEqual('humpback+whale', labels[17])
        
    def test_get_attributes_set(self):
        '''
        Tests if list of attributes is retrieved
        '''
        attributes = self.parser.get_attributes_set()
         
        self.assertEqual(85, len(attributes))
        self.assertEqual('orange', attributes[5])
        self.assertEqual('tree', attributes[76])
        self.assertEqual('hands', attributes[19])
        
    def test_get_attributes_subset(self):
        '''
        Tests if list of attributes subset is retrieved
        '''
        attributes = self.parser.get_attributes_subset()
         
        self.assertEqual(23, len(attributes))
        self.assertEqual('orange', attributes[5])
        self.assertEqual('stripes', attributes[10])
        self.assertEqual('hands', attributes[19])
        
    def test_get_attributes_subset_as_dict(self):
        '''
        Tests if dictionary of attributes subset is correctly retrieved
        '''
        attributes = self.parser.get_attributes_subset(as_dict=True)
         
        self.assertTrue(4, len(attributes.keys()))
        self.assertTrue(8, len(attributes['COLOR']))
        self.assertTrue(5, len(attributes['TEXTURE']))
        self.assertTrue(5, len(attributes['SHAPE']))
        self.assertTrue(5, len(attributes['PARTS'])) 
     
    def test_get_predicate_matrix(self):
        '''
        Tests if data frame with predicate matrix is retrieved
        '''
        attributes = self.parser.get_predicate_matrix()
         
        self.assertTrue(isinstance(attributes, pd.DataFrame))
        self.assertEqual(self.parser.get_labels(), list(attributes.index.values))
        self.assertEqual(self.parser.get_attributes_set(), list(attributes.columns.values))
        self.assertEqual((50,), attributes['toughskin'].values.shape)
        self.assertEqual((85,), attributes.loc['gorilla'].values.shape)
        
    def test_get_predicate_matrix_subset(self):
        '''
        Tests if data frame with predicate matrix for subset is retrieved
        '''
        attributes = self.parser.get_predicate_matrix(subset=True)
         
        self.assertTrue(isinstance(attributes, pd.DataFrame))
        self.assertEqual(self.parser.get_labels(), list(attributes.index.values))
        self.assertEqual(self.parser.get_attributes_subset(), list(attributes.columns.values))
        self.assertEqual((50,), attributes['horns'].values.shape)
        self.assertEqual((23,), attributes.loc['gorilla'].values.shape)
         
    def test_attributes_content(self):
        '''
        Tests if data frame with predicate matrix has reasonable values
        '''
        attributes = self.parser.get_predicate_matrix()
         
        for label in self.parser.get_labels():
            self.assertTrue(sum(attributes.loc[label].values) < 85)
            for value in attributes.loc[label].values:
                self.assertTrue(value == 0 or value == 1)
             
        for att in self.parser.get_attributes_set():
            self.assertTrue(sum(attributes[att].values) < 50)
            for value in attributes.loc[label].values:
                self.assertTrue(value == 0 or value == 1)
 
    def test_attributes_content_subset(self):
        '''
        Tests if data frame with predicate matrix or subset has reasonable values
        '''
        attributes = self.parser.get_predicate_matrix(subset=True)
         
        for label in self.parser.get_labels():
            self.assertTrue(sum(attributes.loc[label].values) < 23)
            for value in attributes.loc[label].values:
                self.assertTrue(value == 0 or value == 1)
             
        for att in self.parser.get_attributes_subset():
            self.assertTrue(sum(attributes[att].values) < 50)
            for value in attributes.loc[label].values:
                self.assertTrue(value == 0 or value == 1)
