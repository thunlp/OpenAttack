import random

from .base import WordSubstitute
from ..data_manager import DataManager
from ..exceptions import UnknownPOSException


class ChineseHowNetSubstitute(WordSubstitute):
    """
    :Package Requirements: OpenHowNet
    :Data Requirements: :py:data:`.AttackAssist.HowNet`

    An implementation of :py:class:`.WordSubstitute`.

    Chinese Sememe-based word substitute based OpenHowNet.

    """

    def __init__(self):
        super().__init__()
        self.hownet_dict = DataManager.load("AttackAssist.HowNet")
        self.zh_word_list = self.hownet_dict.get_zh_words()

    def __call__(self, word, pos, threshold=None):
        """
        :param word: the raw word; pos_tag: part of speech of the word, threshold: return top k words.
        :return: The result is a list of tuples, *(substitute, 1)*.
        :rtype: list of tuple
        """
        if pos is  None:
            pp = None
        elif pos[:2] == "JJ":
            pp = "adj"
        elif pos[:2] == "VB":
            pp = "verb"
        elif pos[:2] == "NN":
            pp = "noun"
        elif pos[:2] == "RB":
            pp = "adv"
        else:
            pp = None
        pos_tag = pp
        if pos_tag is None:
            return [word]

        word_candidate = []

        pos_list = ['noun', 'verb', 'adj', 'adv']
        pos_set = set(pos_list)
        if pos_tag not in pos_set:
            raise UnknownPOSException(word, pos_tag)

        # get sememes
        word_sememes = self.hownet_dict.get_sememes_by_word(word, structured=False, lang="zh", merge=False)
        word_sememe_sets = [t['sememes'] for t in word_sememes]
        if len(word_sememes) == 0:
            return [word]

        # find candidates
        for wd in self.zh_word_list:
            if wd is word:
                continue

            # pos
            word_pos = set()
            word_pos.add(pos_tag)
            result_list2 = self.hownet_dict.get(wd)
            wd_pos = set()
            for a in result_list2:
                if type(a) != dict:
                    continue
                wd_pos.add(a['en_grammar'])
            all_pos = wd_pos & word_pos & pos_set
            if len(all_pos) == 0:
                continue

            # sememe
            wd_sememes = self.hownet_dict.get_sememes_by_word(wd, structured=False, lang="zh", merge=False)
            wd_sememe_sets = [t['sememes'] for t in wd_sememes]
            if len(wd_sememes) == 0:
                continue
            can_be_sub = False
            for s1 in word_sememe_sets:
                for s2 in wd_sememe_sets:
                    if s1 == s2:
                        can_be_sub = True
                        break
            if can_be_sub:
                word_candidate.append(wd)

        word_candidate_1 = []
        for wd in word_candidate:
            wdlist = wd.split(' ')
            if len(wdlist) == 1:
                word_candidate_1.append(wd)
        ret = []
        for wd in word_candidate_1:
            ret.append((wd, 1))
        random.shuffle(ret)
        if threshold is not None:
            ret = ret[:threshold]
        return ret
