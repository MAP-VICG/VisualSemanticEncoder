"""
Module for normalizing datasets and saving the maximum and minimum values 
of it before normalization

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Nov 13, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""


class Normalization:

    @staticmethod
    def normalize_zero_one(fts, maximum, minimum):
        """
        Normalizes array of features to values between 0 and 1

        @param fts: numpy 1D array with features
        @param maximum: float number of features maximum value
        @param minimum: float number of features minimum value
        @return: None
        """
        fts -= minimum
        fts /= (maximum - minimum) + 0.00001

    @staticmethod
    def normalize_zero_one_global(fts):
        """
        Normalizes array of features to values between 0 and 1 considering global maximum and minimum values

        @param fts: numpy 2D array with features
        @return: None
        """
        max_global = fts.max()
        min_global = fts.min()
        Normalization.normalize_zero_one(fts, max_global, min_global)

    @staticmethod
    def normalize_zero_one_by_column(fts):
        """
        Normalizes array of features to values between 0 and 1 considering maximum and minimum values by column

        @param fts: numpy 2D array with features
        @return: None
        """
        max_column = [fts[:, col].max() for col in range(fts.shape[1])]
        min_column = [fts[:, col].min() for col in range(fts.shape[1])]

        for col in range(fts.shape[1]):
            Normalization.normalize_zero_one(fts[:, col], max_column[col], min_column[col])
