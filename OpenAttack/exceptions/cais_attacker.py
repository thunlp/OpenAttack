from ..exception import AttackException

class WordEmbeddingRequired(AttackException):
    pass

class TokensNotAligned(AttackException):
    pass