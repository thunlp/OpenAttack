from .base import CharSubstitute
from ..data_manager import DataManager


class ChineseSimCharSubstitute(CharSubstitute):
    """
    :Data Requirements: :py:data:`.AttackAssist.SIM`
    
    An implementation of :py:class:`.CharSubstitute`.
    """

    def __init__(self):
        super().__init__()
        self.sim_dict = DataManager.load("AttackAssist.SIM")
        # load

    def __call__(self, char, threshold=None):
        """
        :param char: the raw char, threshold: return top k words.
        :return: The result is a list of tuples, *(substitute, 1)*.
        :rtype: list of tuple
        """
        print(self.sim_dict)
        if char in self.sim_dict:
            res = list()
            for chr in self.sim_dict[char]:
                res.append((chr, 1))
            if threshold is not None:
                return res[:threshold]
            else:
                return res
        return list()
