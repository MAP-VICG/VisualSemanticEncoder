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
    def __init__(self, fts):
        """
        Initializes main variables

        @param fts: numpy 2D array with features
        """
        self.max_global = fts.max()
        self.min_global = fts.min()
        self.max_column = [fts[:, col].max() for col in range(fts.shape[1])]
        self.min_column = [fts[:, col].min() for col in range(fts.shape[1])]

    @staticmethod
    def _normalize_zero_one(fts, maximum, minimum):
        """
        Normalizes array of features to values between 0 and 1

        @param fts: numpy 1D array with features
        @param maximum: float number of features maximum value
        @param minimum: float number of features minimum value
        @return: None
        """
        fts -= minimum
        fts /= (maximum-minimum) + 1
        
    def normalize_zero_one_global(self, fts):
        """
        Normalizes array of features to values between 0 and 1 considering global maximum and minimum values

        @param fts: numpy 2D array with features
        @return: None
        """
        self._normalize_zero_one(fts, self.max_global, self.min_global)

    def normalize_zero_one_by_column(self, fts):
        """
        Normalizes array of features to values between 0 and 1 considering maximum and minimum values by column

        @param fts: numpy 2D array with features
        @return: None
        """
        for col in range(fts.shape[1]):
            self._normalize_zero_one(fts[:, col], self.max_column[col], self.min_column[col])
