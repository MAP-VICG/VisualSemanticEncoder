"""
Tests for module matlabparser

@author: Damares Resende
@contact: damaresresende@usp.br
@since: June 15, 2020

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""
import unittest
from scipy.io import loadmat

from featureextraction.src.matlaparser import Parser


class MatlabParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Loads data
        """
        cls.awa_prs = Parser('awa')
        x_vis = '../../../Datasets/AWA2/AWA2_x_vis.txt'
        x_sem = '../../../Datasets/AWA2/AWA2_x_sem.txt'
        y = '../../../Datasets/AWA2/AWA2_y.txt'
        cls.awa_vis_data, cls.awa_sem_data, cls.awa_labels = cls.awa_prs.load_awa_data(x_vis, x_sem, y)

        cls.cub_prs = Parser('cub')
        tr_vis = '../../../Datasets/CUB200/CUB200_x_train_vis.txt'
        te_vis = '../../../Datasets/CUB200/CUB200_x_test_vis.txt'
        tr_sem = '../../../Datasets/CUB200/CUB200_x_train_sem.txt'
        te_sem = '../../../Datasets/CUB200/CUB200_x_test_sem.txt'
        tr_y = '../../../Datasets/CUB200/CUB200_y_train.txt'
        te_y = '../../../Datasets/CUB200/CUB200_y_test.txt'
        cls.cub_vis_data, cls.cub_sem_data, cls.cub_labels = cls.cub_prs.load_cub_data(tr_vis, te_vis, tr_sem, te_sem,
                                                                                       tr_y, te_y)

        cls.template_cub = loadmat('../../../Datasets/SAE/cub_demo_data.mat')
        cls.template_awa = loadmat('../../../Datasets/SAE/awa_demo_data.mat')

    def test_type_setting_cub(self):
        """
        Tests if test labels are correctly set to cub data type
        """
        template = [x[0] for x in self.template_cub['te_cl_id']]

        self.assertEqual(template, self.cub_prs.test_labels)

    def test_type_setting_awa(self):
        """
        Tests if test labels are correctly set to awa data type
        """
        template = [x[0] for x in self.template_awa['param']['testclasses_id'][0][0]]

        self.assertEqual(template, self.awa_prs.test_labels)

    def test_type_setting_error(self):
        """
        Tests if test labels settings throw an error
        """
        self.assertRaises(ValueError, Parser, 'smt')

    def test_build_masks_awa(self):
        """
        Tests if masks arrays are correctly constructed for awa
        """
        tr_mask, te_mask = self.awa_prs._build_masks(self.awa_labels)
        self.assertEqual(37322, len(tr_mask))
        self.assertEqual(37322, len(te_mask))
        self.assertEqual(6985, sum([1 for v in te_mask if v]))
        self.assertEqual(30337, sum([1 for v in te_mask if not v]))
        self.assertEqual(30337, sum([1 for v in tr_mask if v]))
        self.assertEqual(6985, sum([1 for v in tr_mask if not v]))

    def test_build_masks_cub(self):
        """
        Tests if masks arrays are correctly constructed for cub
        """
        tr_mask, te_mask = self.cub_prs._build_masks(self.cub_labels)
        self.assertEqual(11788, len(tr_mask))
        self.assertEqual(11788, len(te_mask))
        self.assertEqual(2933, sum([1 for v in te_mask if v]))
        self.assertEqual(8855, sum([1 for v in te_mask if not v]))
        self.assertEqual(8855, sum([1 for v in tr_mask if v]))
        self.assertEqual(2933, sum([1 for v in tr_mask if not v]))

    def test_split_data_awa(self):
        """
        Tests if when data is split, the resulting dictionary is built as expected for awa
        """
        self.awa_prs.split_data(self.awa_vis_data, self.awa_sem_data, self.awa_labels)
        self.assertEqual(['S_te', 'S_te_gt', 'S_te_pro', 'S_tr', 'X_te', 'X_tr', 'param'],
                         sorted(list(self.awa_prs.data_dict.keys())))
        self.assertEqual(['test_labels', 'testclasses_id', 'train_labels'],
                         sorted(list(self.awa_prs.data_dict['param'].keys())))
        self.assertEqual((6985, 85), self.awa_prs.data_dict['S_te'].shape)
        self.assertEqual((30337, 85), self.awa_prs.data_dict['S_tr'].shape)
        self.assertEqual((6985, 2048), self.awa_prs.data_dict['X_te'].shape)
        self.assertEqual((30337, 2048), self.awa_prs.data_dict['X_tr'].shape)
        self.assertEqual((6985, 1), self.awa_prs.data_dict['param']['test_labels'].shape)
        self.assertEqual((10, 1), self.awa_prs.data_dict['param']['testclasses_id'].shape)
        self.assertEqual((30337, 1), self.awa_prs.data_dict['param']['train_labels'].shape)

    def test_split_data_cub(self):
        """
        Tests if when data is split, the resulting dictionary is built as expected for cub
        """
        self.cub_prs.split_data(self.cub_vis_data, self.cub_sem_data, self.cub_labels)
        self.assertEqual(['S_te', 'S_te_pro', 'S_tr', 'X_te', 'X_tr', 'te_cl_id', 'test_labels_cub',
                          'train_labels_cub'], sorted(list(self.cub_prs.data_dict.keys())))
        self.assertEqual((2933, 312), self.cub_prs.data_dict['S_te'].shape)
        self.assertEqual((8855, 312), self.cub_prs.data_dict['S_tr'].shape)
        self.assertEqual((2933, 2048), self.cub_prs.data_dict['X_te'].shape)
        self.assertEqual((8855, 2048), self.cub_prs.data_dict['X_tr'].shape)
        self.assertEqual((50, 1), self.cub_prs.data_dict['te_cl_id'].shape)
        self.assertEqual((2933, 1), self.cub_prs.data_dict['test_labels_cub'].shape)
        self.assertEqual((8855, 1), self.cub_prs.data_dict['train_labels_cub'].shape)

    def test_load_awa(self):
        """
        Tests if data loaded is in the expected shape for awa
        """
        awa_prs = Parser('awa')
        x_vis = '../../../Datasets/AWA2/AWA2_x_vis.txt'
        x_sem = '../../../Datasets/AWA2/AWA2_x_sem.txt'
        y = '../../../Datasets/AWA2/AWA2_y.txt'
        awa_vis_data, awa_sem_data, awa_labels = awa_prs.load_awa_data(x_vis, x_sem, y)
        self.assertEqual((37322, 2048), awa_vis_data.shape)
        self.assertEqual((37322, 85), awa_sem_data.shape)
        self.assertEqual((37322,), awa_labels.shape)

    def test_load_cub(self):
        """
        Tests if data loaded is in the expected shape for cub
        """
        cub_prs = Parser('cub')
        tr_vis = '../../../Datasets/CUB200/CUB200_x_train_vis.txt'
        te_vis = '../../../Datasets/CUB200/CUB200_x_test_vis.txt'
        tr_sem = '../../../Datasets/CUB200/CUB200_x_train_sem.txt'
        te_sem = '../../../Datasets/CUB200/CUB200_x_test_sem.txt'
        tr_y = '../../../Datasets/CUB200/CUB200_y_train.txt'
        te_y = '../../../Datasets/CUB200/CUB200_y_test.txt'
        cub_vis_data, cub_sem_data, cub_labels = cub_prs.load_cub_data(tr_vis, te_vis, tr_sem, te_sem, tr_y, te_y)
        self.assertEqual((11788, 2048), cub_vis_data.shape)
        self.assertEqual((11788, 312), cub_sem_data.shape)
        self.assertEqual((11788,), cub_labels.shape)

    def test_build_semantic_matrix_cub(self):
        """
        Tests if the semantic matrix matches the template one for cub
        """
        cub_matrix = self.cub_prs.build_semantic_matrix(self.cub_sem_data, self.cub_labels)
        self.assertEqual((50, 312), cub_matrix.shape)
        self.assertTrue((self.template_cub['S_te_pro'] == cub_matrix).all())

    def test_build_semantic_matrix_awa(self):
        """
        Tests if the semantic matrix matches the template one for awa
        """
        awa_matrix = self.awa_prs.build_semantic_matrix(self.awa_sem_data, self.awa_labels)
        self.assertEqual((10, 85), awa_matrix.shape)
        self.assertTrue((self.template_awa['S_te_pro'] == awa_matrix).all())
