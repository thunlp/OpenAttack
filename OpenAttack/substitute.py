import abc


class Substitute(metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def __call__(self, word_or_char, **kwargs):
        pass
