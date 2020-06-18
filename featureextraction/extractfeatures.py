"""
Extract features from images and merge them with its respective semantic data

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Jan 12, 2020

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""
import sys
import time
from os import path

from featureextraction.src.matlaparser import Parser
from featureextraction.src.dataparsing import CUB200Data, AWA2Data, DataIO


def extract_features(data_type, src_dir, dst_dir):
    """
    Runs ResNet50 model in the input data to extract its features

    :param data_type: type of data: AWA2 or CUB200
    :param src_dir: string with path to the images to load
    :param dst_dir: string with path to where to save the extracted features
    :return: None
    """
    if data_type == 'CUB200':
        data = CUB200Data(src_dir)
        print('Extracting data from CUB200')
    elif data_type == 'AWA2':
        data = AWA2Data(src_dir)
        print('Extracting data from AWA2')
    else:
        raise ValueError('Wrong value for data set type. Please choose CUB200 or AWA2.')

    vis_fts, sem_fts, labels = data.build_data()
    if data_type == 'AWA2':
        DataIO.save_files(dst_dir, data_type, x_train_vis=vis_fts, x_train_sem=sem_fts, y_train=labels)
    elif data_type == 'CUB200':
        train_mask, test_mask = data.get_train_test_mask()
        DataIO.save_files(dst_dir, data_type, x_train_vis=vis_fts[train_mask], x_test_vis=vis_fts[test_mask],
                          x_train_sem=sem_fts[train_mask], x_test_sem=sem_fts[test_mask],
                          y_train=labels[train_mask], y_test=labels[test_mask])


def parse_data(data_type, dst_dir):
    """
    Parses the features extracted and saved in a .txt file to a .mat data sttructure

    :param data_type: type of data: AWA2 or CUB200
    :param dst_dir: string with path to where data was saved in te extraction. It is also where the
    parsed data will be saved
    :return: None
    """
    prs = Parser(data_type)
    print('Parsing data from %s' % data_type)

    vis_data, sem_data, labels = prs.load_data(dst_dir)
    prs.split_data(vis_data, sem_data, labels)
    prs.save_data(path.join(dst_dir, '%s_demo_data.mat' % data_type))
    print('Data saved under %s' % path.join(dst_dir, '%s_demo_data.mat' % data_type))


def main():
    init_time = time.time()

    if len(sys.argv) < 3:
        raise IndexError('Please provide input for data set type, source path and destination path')

    extract_features(sys.argv[1], sys.argv[2], sys.argv[3])
    parse_data(sys.argv[1], sys.argv[3])

    elapsed = time.time() - init_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    time_elapsed = '{:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds)

    print('Elapsed time is %s' % time_elapsed)


if __name__ == '__main__':
    main()
