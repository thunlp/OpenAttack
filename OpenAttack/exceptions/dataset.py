from ..exception import AttackException

class DuplicatedParameterException(AttackException):
    pass

class UnknownDataLabelException(AttackException):
    pass

class UnknownDataFormatException(AttackException):
    pass