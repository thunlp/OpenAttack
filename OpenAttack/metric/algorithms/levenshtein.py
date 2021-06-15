from .base import AttackMetric
import torch
from ...text_process.tokenizer import Tokenizer

class Levenshtein(AttackMetric):
    
    NAME = "Levenshtein Edit Distance"

    def __init__(self, tokenizer : Tokenizer) -> None:
        self.tokenizer = tokenizer

    @property
    def TAGS(self):
        if hasattr(self.tokenizer, "TAGS"):
            return self.tokenizer.TAGS
        return set()
        
    def calc_score(self, a, b):
        """
            :param list a: The first list.
            :param list b: The second list.

            Both parameters can be str or list, str for char-level edit distance while list for token-level edit distance.
            """
        la = len(a)
        lb = len(b)
        f = torch.zeros(la + 1, lb + 1, dtype=torch.long)
        for i in range(la + 1):
            for j in range(lb + 1):
                if i == 0:
                    f[i][j] = j
                elif j == 0:
                    f[i][j] = i
                elif a[i - 1] == b[j - 1]:
                    f[i][j] = f[i - 1][j - 1]
                else:
                    f[i][j] = min(f[i - 1][j - 1], f[i - 1][j], f[i][j - 1]) + 1
        return f[la][lb]

    def after_attack(self, input, adversarial_sample):
        if adversarial_sample is not None:
            return self.calc_score( self.tokenizer.tokenize(input["x"], pos_tagging=False), self.tokenizer.tokenize(adversarial_sample, pos_tagging=False) )