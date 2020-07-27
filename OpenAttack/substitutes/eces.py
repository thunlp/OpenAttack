"""
    对26个字母（大小写），返回一个最相近字母，同时希望其引起最大影响
    进一步工作：对一段文本以概率p进行扰动
"""
from .base import CharSubstitute
# from ..exceptions import


H = {'a': 'â', 'b': 'ḃ', 'c': 'ĉ', 'd': 'ḑ', 'e': 'ê', 'f': 'ḟ', 'g': 'ǵ', 'h': 'ĥ', 'i': 'î',
     'j': 'ĵ', 'k': 'ǩ', 'l': 'ᶅ', 'm': 'ḿ', 'n': 'ň', 'o': 'ô', 'p': 'ṕ', 'q': 'ʠ', 'r': 'ř',
     's': 'ŝ', 't': 'ẗ', 'u': 'ǔ', 'v': 'ṽ', 'w': 'ẘ', 'x': 'ẍ', 'y': 'ŷ', 'z': 'ẑ',
     'A': 'Â', 'B': 'Ḃ', 'C': 'Ĉ', 'D': 'Ď', 'E': 'Ê', 'F': 'Ḟ', 'G': 'Ĝ', 'H': 'Ĥ', 'I': 'Î',
     'J': 'Ĵ', 'K': 'Ǩ', 'L': 'Ĺ', 'M': 'Ḿ', 'N': 'Ň', 'O': 'Ô', 'P': 'Ṕ', 'Q': 'Q', 'R': 'Ř',
     'S': 'Ŝ', 'T': 'Ť', 'U': 'Û', 'V': 'Ṽ', 'W': 'Ŵ', 'X': 'Ẍ', 'Y': 'Ŷ', 'Z': 'Ẑ'}


class ECESSubstitute(CharSubstitute):
    """
    An implementation of :py:class:`.CharSubstitute`.

    ECES substitute used in :py:class:`.VIPERAttacker`.
    """

    def __init__(self):
        self.h = H

    def __call__(self, char, threshold=None):
        if char not in self.h:
            return char
        ret = []
        ret.append((self.h[char], 1))
        return ret
