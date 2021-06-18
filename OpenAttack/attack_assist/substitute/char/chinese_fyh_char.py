from typing import Optional
from .base import CharSubstitute
from ....data_manager import DataManager
from ....tags import *


class ChineseFYHCharSubstitute(CharSubstitute):

    TAGS = { TAG_Chinese }

    def __init__(self, k : Optional[int] = None):
        """
        Returns traditional, variant and Martian characters of the input character.

        Args:
            k: Top-k results to return. If k is `None`, all results will be returned.
        
        :Data Requirements: :py:data:`.AttackAssist.FYH`
        :Language: chinese
        
        """

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
