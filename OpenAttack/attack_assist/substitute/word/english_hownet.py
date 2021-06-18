from .base import WordSubstitute
from ....data_manager import DataManager
from ....exceptions import WordNotInDictionaryException
from ....tags import TAG_English
import pickle


class HowNetSubstitute(WordSubstitute):

    TAGS = { TAG_English }

    def __init__(self, k = None):
        """
        English Sememe-based word substitute based on OpenHowNet.
        `[pdf] <https://arxiv.org/pdf/1901.09957.pdf>`__

        Args:
            k: Top-k results to return. If k is `None`, all results will be returned.
        
        :Data Requirements: :py:data:`.AttackAssist.HownetSubstituteDict`
        :Language: english
        
        """

        with open(DataManager.load("AttackAssist.HownetSubstituteDict"),'rb') as fp:
            self.dict=pickle.load(fp)
        self.k = k

    def substitute(self, word: str, pos: str):

        if word not in self.dict or pos not in self.dict[word]:
            raise WordNotInDictionaryException()

        word_candidate = self.dict[word][pos]
        
        ret = []
        for wd in word_candidate:
            ret.append((wd, 1))

        if self.k is not None:
            ret = ret[ : self.k]
        return ret
