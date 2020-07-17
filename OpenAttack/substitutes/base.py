from ..substitute import Substitute
import abc

class WordSubstitute(Substitute):
    @abc.abstractmethod
    def __call__(self, word, pos, **kwargs):
        """
        :param word: the raw word; pos: part of speech of the word.
        :return: The result is a list of tuples, *(substitute, distance)*.
        :rtype: list of tuple
        """
        pass

class CharSubstitute(Substitute):
    @abc.abstractmethod
    def __call__(self, char, **kwargs):
        """
        :param char: the raw char
        :return: The result is a list of tuples, *(substitute, distance)*.
        :rtype: list of tuple
        """
        pass