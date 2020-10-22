"""
Extract features from images and merge them with its respective semantic data

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Jan 12, 2020

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""
import time
import logging
import argparse
from distutils.util import strtobool

from featureextraction.src.fetureextraction import ExtractionType
from featureextraction.src.dataparsing import DataParserFactory, DataParserType


def extract_features(data_type, src_dir, dst_file, extractor, vis_data=True):
    """
    Runs ResNet50 model in the input data to extract its features

    :param data_type: type of data: DataParserType
    :param src_dir: string with path to the images to load
    :param dst_file: string with path to where to save the extracted features
    :param extractor: extractor for visual features (resnet or inception)
    :param vis_data: if True calls visual data extractor
    :return: None
    """
    data = DataParserFactory()(data_type, src_dir, extractor)
    data.build_data_structure(dst_file, vis_data)


def main():
    logging.basicConfig(level=logging.DEBUG,
                        filename='extract_features.log',
                        format='%(asctime)s %(levelname)s [%(module)s, %(funcName)s]: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    init_time = time.time()

    prs = argparse.ArgumentParser(description='Extracts semantic and visual features and save results to a .mat file')
    prs.add_argument('--dtype', help='Database type', choices=[t.value for t in DataParserType])
    prs.add_argument('--etype', help='Extractor type', choices=[t.value for t in ExtractionType])
    prs.add_argument('--sdir', help='Folder where database with image and attributes is located')
    prs.add_argument('--rfile', help='Path and name of results file')
    prs.add_argument('--vis', default=True, type=lambda x: bool(strtobool(x)), help='Path and name of results file')

    args = prs.parse_args()
    args.dtype = DataParserType[args.dtype.upper()]
    args.etype = ExtractionType[args.etype.upper()]
    extract_features(args.dtype, args.sdir, args.rfile, args.etype, args.vis)

    elapsed = time.time() - init_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    time_elapsed = '{:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds)

    logging.info('Elapsed time is %s' % time_elapsed)


if __name__ == '__main__':
    main()
