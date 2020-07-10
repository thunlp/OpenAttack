from ..substitute import Substitute

class WordSubstitute(Substitute):
    @abc.abstractmethod
    def __call__(self, word, pos, **kwargs):
        pass

class CharSubstitute(Substitute):
    @abc.abstractmethod
    def __call__(self, char, **kwargs):
        pass