from typing import Optional
from ....exceptions import WordNotInDictionaryException
from .base import WordSubstitute
from ....data_manager import DataManager
from ....tags import *

class ChineseHowNetSubstitute(WordSubstitute):

    TAGS = { TAG_Chinese }

    def __init__(self, k : Optional[int] = None):
        """
        Chinese Sememe-based word substitute based on OpenHowNet.
        `[pdf] <https://arxiv.org/pdf/1901.09957.pdf>`__

        Args:
            k: Top-k results to return. If k is `None`, all results will be returned.
        
        :Package Requirements: OpenHowNet
        :Data Requirements: :py:data:`.AttackAssist.HowNet`
        :Language: chinese
        
        """

        super().__init__()
        self.hownet_dict = DataManager.load("AttackAssist.HowNet")
        self.zh_word_list = self.hownet_dict.get_zh_words()
        self.k = k

    def substitute(self, word, pos):
        # get sememes
        word_sememes = self.hownet_dict.get_sememes_by_word(word, structured=False, lang="zh", merge=False)
        word_sememe_set = [t['sememes'] for t in word_sememes]
        if len(word_sememes) == 0:
            raise WordNotInDictionaryException()

        # find candidates
        word_candidate = [(word, 0)]
        for wd in self.zh_word_list:
            if wd == word:
                continue

            wd_pos = set()
            for a in self.hownet_dict.get(wd):
                if type(a) is not dict:
                    continue
                wd_pos.add(a['en_grammar'])
            if pos not in wd_pos:
                continue

            # sememe
            wd_sememes = self.hownet_dict.get_sememes_by_word(wd, structured=False, lang="zh", merge=False)
            wd_sememe_set = [t['sememes'] for t in wd_sememes]
            if len(wd_sememes) == 0:
                continue
            
            common_sememe = 0
            for s1 in word_sememe_set:
                for s2 in wd_sememe_set:
                    if s1 == s2:
                        common_sememe += 1
            
            if common_sememe > 0:
                if wd.find(" ") == -1:
                    word_candidate.append((wd, 1 - common_sememe / len(word_sememe_set)) )

        word_candidate = sorted(word_candidate, key=lambda x: x[1])
        if self.k is not None:
            word_candidate = word_candidate[:self.k]
        return word_candidate
