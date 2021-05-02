from .base import CharSubstitute
from ..data_manager import DataManager


class ChineseFYHCharSubstitute(CharSubstitute):
    """
    :Data Requirements: :py:data:`.AttackAssist.FYH`
    
    An implementation of :py:class:`.CharSubstitute`.
    """

    def __init__(self):
        super().__init__()
        self.tra_dict, self.var_dict, self.hot_dict = DataManager.load("AttackAssist.FYH")
        # load

    def __call__(self, char, threshold=None):
        """
        :param char: the raw char, threshold: return top k words.
        :return: The result is a list of tuples, *(substitute, 1)*.
        :rtype: list of tuple
        :Data Requirements: :py:data:`.AttackAssist.FYH`
        """
        if char in self.tra_dict or char in self.var_dict or char in self.hot_dict:
            fanyihuo_result = set()
            if char in self.tra_dict:
                fanyihuo_result = fanyihuo_result.union(self.tra_dict[char])
            if char in self.var_dict:
                fanyihuo_result = fanyihuo_result.union(self.var_dict[char])
            if char in self.hot_dict:
                fanyihuo_result = fanyihuo_result.union(self.hot_dict[char])
            res = list()
            for ch in fanyihuo_result:
                res.append((ch, 1))
            if threshold is not None:
                return res[:threshold]
            else:
                return res
        return list()
