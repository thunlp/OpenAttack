"""
    对26个字母（大小写），返回一个最相近字母，同时希望其引起最大影响
    进一步工作：对一段文本以概率p进行扰动
"""
from .base import CharSubstitute
from ....tags import *


H = {'a': 'â', 'b': 'ḃ', 'c': 'ĉ', 'd': 'ḑ', 'e': 'ê', 'f': 'ḟ', 'g': 'ǵ', 'h': 'ĥ', 'i': 'î',
     'j': 'ĵ', 'k': 'ǩ', 'l': 'ᶅ', 'm': 'ḿ', 'n': 'ň', 'o': 'ô', 'p': 'ṕ', 'q': 'ʠ', 'r': 'ř',
     's': 'ŝ', 't': 'ẗ', 'u': 'ǔ', 'v': 'ṽ', 'w': 'ẘ', 'x': 'ẍ', 'y': 'ŷ', 'z': 'ẑ',
     'A': 'Â', 'B': 'Ḃ', 'C': 'Ĉ', 'D': 'Ď', 'E': 'Ê', 'F': 'Ḟ', 'G': 'Ĝ', 'H': 'Ĥ', 'I': 'Î',
     'J': 'Ĵ', 'K': 'Ǩ', 'L': 'Ĺ', 'M': 'Ḿ', 'N': 'Ň', 'O': 'Ô', 'P': 'Ṕ', 'Q': 'Q', 'R': 'Ř',
     'S': 'Ŝ', 'T': 'Ť', 'U': 'Û', 'V': 'Ṽ', 'W': 'Ŵ', 'X': 'Ẍ', 'Y': 'Ŷ', 'Z': 'Ẑ'}


class ECESSubstitute(CharSubstitute):

    TAGS = { TAG_English }

    def __init__(self):
        """
        Returns the chars that is visually similar to the input.

        DCES substitute used in :py:class:`.VIPERAttacker`.

        :Data Requirements: :py:data:`.AttackAssist.SIM`
        :Language: english

        """
        self.h = H

    def substitute(self, char: str):
        """
        :param word: the raw char, threshold: return top k chars.
        :return: The result is a list of tuples, *(substitute, 1)*.
        :rtype: list of tuple
        """
        if char not in self.h:
            return [(char, 1)]
        return [(self.h[char], 1)]
