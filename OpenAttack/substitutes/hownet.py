from .base import WordSubstitute
from ..data_manager import DataManager
from ..exceptions import UnknownPOSException


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
        self.hownet_dict = DataManager.load("AttackAssist.HowNet")
        self.wn = DataManager.load("TProcess.NLTKWordNet")
        self.en_word_list = self.hownet_dict.get_en_words()

    def __call__(self, word, pos_tag, threshold=None):
        pp = "noun"
        if pos_tag[:2] == "JJ":
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
        word_candidate = []
        if pos_tag not in pos_set:
            raise UnknownPOSException(word, pos_tag)

        if pos_tag == 'adv':
            word_origin = self.wn.lemma(word, pos='r')
        else:
            word_origin = self.wn.lemma(word, pos=pos_tag[0])

        # pos tagging
        result_list = self.hownet_dict.get(word_origin)
        word_pos = set()
        word_pos.add(pos_tag)

        # get sememes
        word_sememes = self.hownet_dict.get_sememes_by_word(word_origin, structured=False, lang="en", merge=False)
        word_sememe_sets = [t['sememes'] for t in word_sememes]
        if len(word_sememes) == 0:
            return [word]

        # find candidates
        for wd in self.en_word_list:
            if wd == word_origin:
                continue

            # POS
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
            wd_sememes = self.hownet_dict.get_sememes_by_word(wd, structured=False, lang="en", merge=False)
            wd_sememe_sets = [t['sememes'] for t in wd_sememes]
            if len(wd_sememes) == 0:
                continue
            can_be_sub = False
            for s1 in word_sememe_sets:
                for s2 in wd_sememe_sets:
                    if s1 == s2:
                        can_be_sub = True
                        break
            if can_be_sub:  # have same sememes
                for pos_valid in all_pos:
                    if pos_valid == pos_tag:
                        word_candidate.append(wd)

        word_candidate_1 = []
        for wd in word_candidate:
            wdlist = wd.split(' ')
            if len(wdlist) == 1:
                word_candidate_1.append(wd)
        ret = []
        for wd in word_candidate_1:
            ret.append((wd, 1))
        return ret  # todo: rank by same sememes
