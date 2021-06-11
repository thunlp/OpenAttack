from .base import WordSubstitute
from ....data_manager import DataManager
from ....exceptions import WordNotInDictionaryException
from ....tags import *

class ChineseCiLinSubstitute(WordSubstitute):
    """
    :Data Requirements: :py:data:`.AttackAssist.CiLin`

    An implementation of :py:class:`.WordSubstitute`.

    Chinese Sememe-based word substitute based CiLin.
    """
    TAGS = { TAG_Chinese }

    def __init__(self, k = None):
        """
        :param k: Return top `k` candidate words. (return all if k = None)
        """
        self.k = k
        self.cilin_dict = DataManager.load("AttackAssist.CiLin")

    def substitute(self, word, pos_tag):
        """
        :param word: the raw word; 
        :param pos_tag: part of speech of the word
        :return: The result is a list of tuples, *(substitute, 1)*.
        :rtype: list of tuple
        """
        if word not in self.cilin_dict:
            raise WordNotInDictionaryException()
        sym_words = self.cilin_dict[word]
        
        ret = []
        for sym_word in sym_words:
            ret.append((sym_word, 1))

        if self.k is not None:
            ret = ret[:self.k]
        return ret
