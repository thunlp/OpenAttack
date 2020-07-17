import abc


class Substitute(metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def __call__(self, word_or_char, **kwargs):
        """
        :param word_or_char: the raw word or char
        :return: The result is a list of tuples, *(substitute, distance)*.
        :rtype: list of tuple
        """
        pass
