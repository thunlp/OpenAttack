from .base import WordSubstitute
from ..data_manager import DataManager
from ..exceptions import UnknownPOSException
import pickle

pos_list = ['noun', 'verb', 'adj', 'adv']
pos_set = set(pos_list)


class HowNetSubstitute(WordSubstitute):
    """
    :Package Requirements: OpenHowNet
    :Data Requirements: :py:data:`.AttackAssist.HowNet` :py:data:`.TProcess.NLTKWordNet`

    An implementation of :py:class:`.WordSubstitute`.

    Sememe-based word substitute based OpenHowNet.

    """

    def __init__(self):
        with open(DataManager.load("AttackAssist.HownetSubstituteDict"),'rb') as fp:
            self.dict=pickle.load(fp)

    def __call__(self, word, pos_tag, threshold=None):
        """
        :param word: the raw word; pos_tag: part of speech of the word, threshold: return top k words.
        :return: The result is a list of tuples, *(substitute, 1)*.
        :rtype: list of tuple
        """
        pp = "noun"
        if pos_tag is None:
            pp = None
        elif pos_tag[:2] == "JJ":
            pp = "adj"
        elif pos_tag[:2] == "VB":
            pp = "verb"
        elif pos_tag[:2] == "NN":
            pp = "noun"
        elif pos_tag[:2] == "RB":
            pp = "adv"
        else:
            pp = None
        pos_tag = pp
        if pos_tag is None:
            return [word]
        word_candidate_1 = []
        if pos_tag not in pos_set:
            raise UnknownPOSException(word, pos_tag)
        word_candidate_1=self.dict[word][pos_tag]
        ret=[]
        for wd in word_candidate_1:
            ret.append((wd, 1))
        return ret  # todo: rank by same sememes
