"""
Extract features from images and merge them with its respective semantic data

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Jan 12, 2020

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""
import os

from featureextraction.src.dataparsing import BirdsData

DATA_DIR = os.path.expanduser(os.path.join('~', 'Projects', 'Datasets', 'CUB200'))
CUB_DIR = os.path.join(DATA_DIR, 'CUB_200_2011')

data = BirdsData(CUB_DIR)
train_fts, train_class, test_fts, test_class = data.build_birds_data()
data.save_files(DATA_DIR, train_fts, train_class, test_fts, test_class)
