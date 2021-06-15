from .base import AttackMetric
from ...tags import *

class JaccardChar(AttackMetric):

    NAME = "Jaccard Char Similarity"
    TAGS = { * TAG_ALL_LANGUAGE }

    def calc_score(self, senA, senB):
        """
            :param list tokenA: The first list of tokens.
            :param list tokenB: The second list of tokens.

            Make sure two list have the same length.
        """
        AS=set()
        BS=set()
        for i in range(len(senA)):
            AS.add(senA[i])
        for i in range(len(senB)):
            BS.add(senB[i])

        return len(AS&BS)/len(AS|BS)
    
    def after_attack(self, input, adversarial_sample):
        if adversarial_sample is not None:
            return self.calc_score( input["x"], adversarial_sample )
        return None
    