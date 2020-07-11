from ..exception import AttackException


class NoEmbeddingException(AttackException):
    pass


class WordNotInDictionaryException(AttackException):
    pass

class UnknownPOSException(AttackException):
    pass