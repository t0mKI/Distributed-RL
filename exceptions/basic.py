from abc import ABC, abstractmethod

class BasicException(ABC, Exception):
    '''
    Abstract class providing basic information for exceptions
    '''

    def __init__(self, msg):
        super().__init__(msg)

    @abstractmethod
    def getDetails(self):
        pass

    @abstractmethod
    def getErrorString(self):
        pass
