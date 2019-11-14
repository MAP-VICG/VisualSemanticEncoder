'''
Singleton base class

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Nov 13, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
'''

class Singleton(type):
    '''
    Singleton base class to be used as a meta class
    '''
    _instances = {}
    def __call__(self, *args, **kwargs):
        if self not in self._instances:
            self._instances[self] = super(Singleton, self).__call__(*args, **kwargs)
        return self._instances[self]