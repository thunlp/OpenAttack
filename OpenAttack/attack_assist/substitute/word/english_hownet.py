from .base import WordSubstitute
from ....data_manager import DataManager
from ....exceptions import WordNotInDictionaryException
from ....tags import TAG_English
import pickle


class HowNetSubstitute(WordSubstitute):
    """
    :Package Requirements: OpenHowNet
    :Data Requirements: :py:data:`.AttackAssist.HowNet` :py:data:`.TProcess.NLTKWordNet`

    An implementation of :py:class:`.WordSubstitute`.

    Sememe-based word substitute based OpenHowNet.

    """

    TAGS = { TAG_English }

    def __init__(self, k = None):
        with open(DataManager.load("AttackAssist.HownetSubstituteDict"),'rb') as fp:
            self.dict=pickle.load(fp)
        self.k = k

    def substitute(self, word: str, pos: str):
        """
        :param word: the raw word; pos_tag: part of speech of the word, threshold: return top k words.
        :return: The result is a list of tuples, *(substitute, 1)*.
        :rtype: list of tuple
        """
       
        if word not in self.dict or pos not in self.dict[word]:
            raise WordNotInDictionaryException()

        word_candidate = self.dict[word][pos]
        
        ret = []
        for wd in word_candidate:
            ret.append((wd, 1))

        if self.k is not None:
            ret = ret[ : self.k]
        return ret
