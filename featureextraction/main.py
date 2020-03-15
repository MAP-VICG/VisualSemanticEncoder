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

from featureextraction.src.dataparsing import CUB200Data, AWA2Data, DataIO


def main():
    init_time = time.time()

    if len(sys.argv) < 3:
        raise IndexError('Please provide input for data set type, source path and destination path')

    data_type = sys.argv[1]
    src_dir = sys.argv[2]
    dst_dir = sys.argv[3]

    if data_type == 'CUB200':
        data = CUB200Data(src_dir)
        print('Extracting data from CUB200')
    elif data_type == 'AWA2':
        data = AWA2Data(src_dir)
        print('Extracting data from AWA2')
    else:
        raise ValueError('Wrong value for data set type. Please choose CUB200 or AWA2.')

    train_fts, test_fts, train_class, test_class = data.build_data()
    DataIO.save_files(dst_dir, train_fts, test_fts, train_class, test_class, data_type)

    elapsed = time.time() - init_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    time_elapsed = '{:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds)

    print('Elapsed time is %s' % time_elapsed)


if __name__ == '__main__':
    main()