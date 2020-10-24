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
        :param int threshold: Returns top k results (k = threshold).
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
