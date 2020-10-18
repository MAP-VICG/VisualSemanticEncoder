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

from featureextraction.src.dataparsing import CUB200Data, AWA2Data, PascalYahooData, SUNData, DataIO


def extract_features(data_type, src_dir, dst_dir, extractor):
    """
    Runs ResNet50 model in the input data to extract its features

    :param data_type: type of data: AWA2, CUB200 or aP&Y
    :param src_dir: string with path to the images to load
    :param dst_dir: string with path to where to save the extracted features
    :param extractor: extractor for visual features (resnet or inception)
    :return: None
    """
    if data_type == 'CUB200':
        data = CUB200Data(src_dir, extractor)
        print('Extracting data from CUB200')
    elif data_type == 'AWA2':
        data = AWA2Data(src_dir, extractor)
        print('Extracting data from AWA2')
    elif data_type == 'aP&Y':
        data = PascalYahooData(src_dir, extractor)
        print('Extracting data from aP&Y')
    elif data_type == 'SUN':
        data = SUNData(src_dir, extractor)
        print('Extracting data from SUN')
    else:
        raise ValueError('Wrong value for data set type. Please choose CUB200, AWA2, aP&Y or SUN.')

    vis_fts, sem_fts, labels = data.build_data()
    if data_type == 'CUB200':
        train_mask, test_mask = data.get_train_test_mask()
        DataIO.save_files(dst_dir, data_type, x_train_vis=vis_fts[train_mask], x_test_vis=vis_fts[test_mask],
                          x_train_sem=sem_fts[train_mask], x_test_sem=sem_fts[test_mask],
                          y_train=labels[train_mask], y_test=labels[test_mask])
    else:
        DataIO.save_files(dst_dir, data_type, x_train_vis=vis_fts, x_train_sem=sem_fts, y_train=labels)


def main():
    init_time = time.time()

    if len(sys.argv) < 4:
        raise IndexError('Please provide input for dataset type, source path, destination path and extractor')

    extract_features(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

    elapsed = time.time() - init_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    time_elapsed = '{:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds)

    print('Elapsed time is %s' % time_elapsed)


if __name__ == '__main__':
    main()
