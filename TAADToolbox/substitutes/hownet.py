"""
    由hownet提供的近义词。
    进一步工作：以义原数量排序
    require:
    DataManager.download("HOWNET")
    DataManager.download("WNL")
"""
from ..substitute import Substitute
from ..data_manager import DataManager
# from ..exceptions import


pos_list = ['noun', 'verb', 'adj', 'adv']
pos_set = set(pos_list)


class HowNetSubstitute(Substitute):

    def __init__(self):
        self.hownet_dict = DataManager.load("HOWNET")
        self.wnl = DataManager.load("WNL")
        self.en_word_list = self.hownet_dict.get_en_words()
        # self.hownet_dict = OpenHowNet.HowNetDict()
        # self.wnl = WordNetLemmatizer()

    def __call__(self, word_or_char, pos_tag):
        word_candidate = []
        # pos_tag = 'noun' 'verb' 'adj' 'adv'
        if pos_tag not in pos_set:
            print("pos should be 'noun' 'verb' 'adj' or 'adv'")
            return

        if pos_tag == 'adv':
            word_origin = self.wnl.lemmatize(word_or_char, pos='r')
        else:
            word_origin = self.wnl.lemmatize(word_or_char, pos=pos_tag[0])

        # pos tagging
        result_list = self.hownet_dict.get(word_origin)
        word_pos = set()
        word_pos.add(pos_tag)
        '''for a in result_list:
            if type(a) != dict:
                continue
            word_pos.add(a['en_grammar'])
        if len(word_pos & pos_set) == 0:
            # raise exception
            return'''

        # get sememes
        word_sememes = self.hownet_dict.get_sememes_by_word(word_origin, structured=False, lang="en", merge=False)
        word_sememe_sets = [t['sememes'] for t in word_sememes]
        if len(word_sememes) == 0:
            # raise exception
            return

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
            # print("w1_sememes: " + word_sememe_sets)
            # print("w2_sememes: " + wd_sememe_sets)
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

        # print("word_candidate:", word_candidate)
        # print("len:", len(word_candidate))
        word_candidate_1 = []
        for wd in word_candidate:
            wdlist = wd.split(' ')
            # print("000", wd, len(wdlist))
            if len(wdlist) == 1:
                # print("111", wd)
                word_candidate_1.append(wd)
        # print("word_candidate_1:", word_candidate_1)
        # print("len:", len(word_candidate_1))
        ret = []
        for wd in word_candidate_1:
            ret.append((wd, 1))
        return ret  # todo: rank by same sememes
