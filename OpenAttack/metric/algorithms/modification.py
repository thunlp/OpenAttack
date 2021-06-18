from typing import List
from .base import AttackMetric
from ...text_process.tokenizer import Tokenizer

class Modification(AttackMetric):
    
    NAME = "Word Modif. Rate"

    def __init__(self, tokenizer : Tokenizer):
        """
        Args:
            tokenizer: A tokenizer that will be used in this metric. Must be an instance of :py:class:`.Tokenizer`

        """
        self.tokenizer = tokenizer

    @property
    def TAGS(self):
        if hasattr(self.tokenizer, "TAGS"):
            return self.tokenizer.TAGS
        return set()
        
    def calc_score(self, tokenA : List[str], tokenB : List[str]) -> float:
        """
        Args:
            tokenA: The first list of tokens.
            tokenB: The second list of tokens.
        Returns:
            Modification rate.

        Make sure two list have the same length.
        """
        va = tokenA
        vb = tokenB
        ret = 0
        if len(va) != len(vb):
            ret = abs(len(va) - len(vb))
        mn_len = min(len(va), len(vb))
        va, vb = va[:mn_len], vb[:mn_len]
        for wordA, wordB in zip(va, vb):
            if wordA != wordB:
                ret += 1
        return ret / len(va)
    
    def after_attack(self, input, adversarial_sample):
        if adversarial_sample is not None:
            return self.calc_score( self.tokenizer.tokenize(input["x"], pos_tagging=False), self.tokenizer.tokenize(adversarial_sample, pos_tagging=False) )
