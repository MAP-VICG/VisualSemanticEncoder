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

from src.utils.singletonbase import Singleton


class MessageType(Enum):
    '''
    Enum for log message type
    '''
    ERR = "ERROR"
    WRN = "WARNING"
    INF = "INFO"


class LogWritter(metaclass=Singleton):
    def __init__(self, logpath=None, console=False):
        '''
        Initializes parameters
        
        @param logpath: if specified, changes log default saving path
        @console console: if True, prints log in console instead of file
        '''
        if logpath and isinstance(logpath, str):
            self.logpath = logpath
        else:
            self.logpath = os.getcwd()
            
        self.console = console
        self.ref_date = datetime.now().strftime('%Y%m%d_%H%M')
    
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
        full_message = '%s\n%s\n%s\n%s: %s\n\n' % ('=' * 100, file_details, execution_details, 
                                                   mtype.value, message)
        
        if self.console:
            print(full_message)
        else:
            with open(log_file, 'a+') as f:
                f.write(full_message)
