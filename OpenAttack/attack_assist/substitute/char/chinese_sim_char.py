from typing import Optional
from .base import CharSubstitute
from ....data_manager import DataManager
from ....tags import *


class ChineseSimCharSubstitute(CharSubstitute):

    TAGS = { TAG_Chinese }

    def __init__(self, k : Optional[int] = None):
        """
        Returns the chars that is visually similar to the input.

        Args:
            k: Top-k results to return. If k is `None`, all results will be returned.
        
        :Data Requirements: :py:data:`.AttackAssist.SIM`
        :Language: chinese
        
        """
        super().__init__()
        self.sim_dict = DataManager.load("AttackAssist.SIM")
        self.k = k

    def substitute(self, char: str):
        ret = []
        if char in self.sim_dict:
            for chr in self.sim_dict[char]:
                ret.append((chr, 1))
        if self.k is not None:
            ret = ret[:self.k]
        return ret
