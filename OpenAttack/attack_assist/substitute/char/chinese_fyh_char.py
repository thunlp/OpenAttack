from .base import CharSubstitute
from ....data_manager import DataManager
from ....tags import *


class ChineseFYHCharSubstitute(CharSubstitute):
    """
    :Data Requirements: :py:data:`.AttackAssist.FYH`
    
    An implementation of :py:class:`.CharSubstitute`.
    """

    TAGS = { TAG_Chinese }

    def __init__(self, k = None):
        super().__init__()
        self.tra_dict, self.var_dict, self.hot_dict = DataManager.load("AttackAssist.FYH")
        self.k = k

    def substitute(self, char: str):
        ret = []
        if char in self.tra_dict or char in self.var_dict or char in self.hot_dict:
            fanyihuo_result = set()
            if char in self.tra_dict:
                fanyihuo_result = fanyihuo_result.union(self.tra_dict[char])
            if char in self.var_dict:
                fanyihuo_result = fanyihuo_result.union(self.var_dict[char])
            if char in self.hot_dict:
                fanyihuo_result = fanyihuo_result.union(self.hot_dict[char])
            for ch in fanyihuo_result:
                ret.append((ch, 1))
            if self.k is not None:
                ret = ret[:self.k]
        return ret
