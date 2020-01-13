"""
Extract features from images and merge them with its respective semantic data

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Jan 12, 2020

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""
from pathlib import Path
from os import path, sep

from featureextraction.src.dataparsing import BirdsData

num_images = 6033
base_path = path.join(str(Path.home()), sep.join(['Projects', 'Datasets', 'Birds']))

data = BirdsData(base_path)
train_fts, train_class, test_fts, test_class = data.build_birds_data(num_images)
data.save_files(base_path, train_fts, train_class, test_fts, test_class)
