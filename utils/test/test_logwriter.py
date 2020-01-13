"""
Tests for module logwriter

@author: Damares Resende
@contact: damaresresende@usp.br
@since: May 3, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""
import unittest
from os import path, remove

from utils.src.logwriter import LogWriter, MessageType


class LogWriterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Initializes writer for all tests
        """
        cls.log = LogWriter()
        cls.log_file = 'semantic_encoder_%s.log' % cls.log.ref_date

    def test_write_message(self):
        self.log.write_message('Testing information messages', MessageType.INF)
        """
        Tests if log file was created with the correct content
        """
        self.assertTrue(path.isfile(self.log_file))

        with open(self.log_file) as f:
            content = ' '.join(f.readlines())

        self.assertTrue('test_write_message' in content)
        self.assertTrue('test_logwriter.py' in content)
        self.assertTrue('INFO: Testing information messages' in content)

    @classmethod
    def tearDownClass(cls):
        """
        Deletes files that were written by the tests
        """
        if path.isfile(cls.log_file):
            remove(cls.log_file)
