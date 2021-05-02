import random
from .base import WordSubstitute
from ..data_manager import DataManager
from ..exceptions import WordNotInDictionaryException


class ChineseCiLinSubstitute(WordSubstitute):
    """
    :Data Requirements: :py:data:`.AttackAssist.CiLin`

    An implementation of :py:class:`.WordSubstitute`.

    Chinese Sememe-based word substitute based CiLin.
    """

    def __init__(self):
        super().__init__()
        self.cilin_dict = DataManager.load("AttackAssist.CiLin")

    def __call__(self, word, pos_tag, threshold=None):
        """
        :param word: the raw word; pos_tag: part of speech of the word, threshold: return top k words.
        :return: The result is a list of tuples, *(substitute, 1)*.
        :rtype: list of tuple
        """
        if word not in self.cilin_dict:
            raise WordNotInDictionaryException()
        sym_words = self.cilin_dict[word]
        ret = []
        for sym_word in sym_words:
            ret.append((sym_word, 1))
        if threshold is not None:
            ret = ret[:threshold]
        return ret
