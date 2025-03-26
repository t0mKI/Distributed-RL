from exceptions.basic import BasicException

class InvalidValueException (BasicException, ValueError):
    '''
    This exception is thrown, when validation failed at a validation method in {@link TTSConfig}
    '''


    def __init__(self, value, detailedMsg, numExplanation,validVals):
        super().__init__("The value \"" + value + "\" is invalid")
        self.detailedMsg = detailedMsg
        self.validVals= numExplanation + "Valid values are: " +  "\n" + validVals  + "."

    def getDetails(self):
        return self.detailedMsg

    def getErrorString(self):
        return self.message + " for the " + self.detailedMsg + "!\n" + self.validVals
