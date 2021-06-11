from .base import CharSubstitute
from ....data_manager import DataManager
from ....tags import *


class ChineseSimCharSubstitute(CharSubstitute):
    """
    :Data Requirements: :py:data:`.AttackAssist.SIM`
    
    An implementation of :py:class:`.CharSubstitute`.
    """

    TAGS = { TAG_Chinese }

    def __init__(self, k = None):
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
