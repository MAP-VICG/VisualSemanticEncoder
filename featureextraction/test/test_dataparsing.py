"""
Tests for module dataparsing

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Jan 5, 2020

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""
import unittest
from os import path, remove
from scipy.io import loadmat
from featureextraction.src.dataparsing import CUB200Data, AWA2Data
from featureextraction.src.fetureextraction import ExtractionType


class AWA2DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_path = path.join('mockfiles', 'AWA2')
        cls.data = AWA2Data(cls.base_path, ExtractionType.RESNET)

    def test_get_visual_attributes(self):
        img_list, _, _ = self.data.get_images_data()
        self.assertEqual((30, 2048), self.data.get_visual_attributes(img_list).shape)

    def test_get_semantic_attributes(self):
        _, img_class, _ = self.data.get_images_data()
        sem_fts, prototypes = self.data.get_semantic_attributes(img_class)
        self.assertEqual((30, 85), sem_fts.shape)
        self.assertEqual((50, 85), prototypes.shape)

    def test_get_images_data(self):
        img_list, img_class, class_dict = self.data.get_images_data()
        img_list.sort()

        self.assertEqual(30, len(img_class))
        self.assertEqual(30, len(img_list))
        self.assertEqual(50, len(class_dict.keys()))

        self.assertEqual([20, 48], img_class[2:4])
        self.assertEqual(20, class_dict['gorilla'])
        self.assertEqual('antelope/antelope_10002.jpg', img_list[0])
        self.assertEqual('wolf/wolf_10094.jpg', img_list[29])

    def test_build_data_structure(self):
        self.data.build_data_structure('dummy_data.mat', vis_data=True)
        img_list, img_class, class_dict = self.data.get_images_data()
        sem_fts, prototypes = self.data.get_semantic_attributes(img_class)

        data = loadmat('dummy_data.mat')
        self.assertEqual(img_class, list(data['img_class'][0]))
        self.assertEqual(img_list, [img.strip() for img in data['img_list']])
        self.assertEqual(class_dict, {v[0].strip(): int(v[1].strip()) for v in data['class_dict']})

        self.assertTrue((data['sem_fts'] == sem_fts).all())
        self.assertTrue((data['prototypes'] == prototypes).all())
        self.assertTrue((data['vis_fts'] == self.data.get_visual_attributes(img_list)).all())

    @classmethod
    def tearDownClass(cls):
        if path.isfile('dummy_data.mat'):
            remove('dummy_data.mat')


class CUB200DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_path = path.join('mockfiles', 'CUB200')
        cls.data = CUB200Data(cls.base_path, ExtractionType.RESNET)

    def test_get_semantic_attributes(self):
        _, img_class, _ = self.data.get_images_data()
        sem_fts, prototypes = self.data.get_semantic_attributes(img_class)
        self.assertEqual((30, 312), sem_fts.shape)
        self.assertEqual((200, 312), prototypes.shape)

    def test_get_images_data(self):
        img_list, img_class, class_dict = self.data.get_images_data()
        img_list.sort()

        self.assertEqual(30, len(img_class))
        self.assertEqual(30, len(img_list))
        self.assertEqual(29, len(class_dict.keys()))

        self.assertEqual([29, 31, 33], img_class[2:5])
        self.assertEqual(200, class_dict['Common_Yellowthroat'])
        self.assertEqual('013.Bobolink/Bobolink_0056_9080.jpg', img_list[0])
        self.assertEqual('112.Great_Grey_Shrike/Great_Grey_Shrike_0050_797012.jpg', img_list[14])
