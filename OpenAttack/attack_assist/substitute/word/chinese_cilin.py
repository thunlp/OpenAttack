from typing import Optional
from .base import WordSubstitute
from ....data_manager import DataManager
from ....exceptions import WordNotInDictionaryException
from ....tags import *

class ChineseCiLinSubstitute(WordSubstitute):
    TAGS = { TAG_Chinese }

    def __init__(self, k : Optional[int] = None):
        """
        Chinese Sememe-based word substitute based CiLin.

        Args:
            k: Top-k results to return. If k is `None`, all results will be returned.
        
        :Data Requirements: :py:data:`.AttackAssist.CiLin`
        :Language: chinese
        
        """

        self.k = k
        self.cilin_dict = DataManager.load("AttackAssist.CiLin")

    def substitute(self, word, pos_tag):
        if word not in self.cilin_dict:
            raise WordNotInDictionaryException()
        sym_words = self.cilin_dict[word]
        
        ret = []
        for sym_word in sym_words:
            ret.append((sym_word, 1))

        if self.k is not None:
            ret = ret[:self.k]
        return ret
