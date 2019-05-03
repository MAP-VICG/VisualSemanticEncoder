'''
Model to write messages in log file

@author: Damares Resende
@contact: damaresresende@usp.br
@since: May 3, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
'''
import os
import inspect
from enum import Enum
from datetime import datetime


class Singleton(type):
    '''
    Singleton base class to be used as a metaclass
    '''
    _instances = {}
    def __call__(self, *args, **kwargs):
        if self not in self._instances:
            self._instances[self] = super(Singleton, self).__call__(*args, **kwargs)
        return self._instances[self]


class MessageType(Enum):
    '''
    Enum for log message type
    '''
    ERR = "ERROR"
    WRN = "WARNING"
    INF = "INFO"


class Logger(metaclass=Singleton):
    def __init__(self, logpath=None):
        '''
        Initializes parameters
        '''
        if logpath:
            self.logpath = logpath
        else:
            self.logpath = os.getcwd()
        self.ref_date = datetime.now().strftime('%Y%m%d_%H%M00')
    
    def write_message(self, message, mtype):
        '''
        Appends message to log file according to the message type. It includes the method call and call
        time in the full message body.
        
        @param message: text message to write
        @param mtype: message type. ERR, WRN or INF
        @return None
        '''
        stack = inspect.stack()
        file_details = stack[1][1] + ' (' + str(stack[1][2]) + ')'
        execution_details = stack[1][3] + '[' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ']'
        
        log_file = os.path.join(self.logpath, 'semantic_encoder_log_%s.log' % self.ref_date)
        
        with open(log_file, 'a+') as f:
            f.write('====================================================================\n%s\n%s\n%s: %s\n\n'
                    % (file_details, execution_details, mtype.value, message))
