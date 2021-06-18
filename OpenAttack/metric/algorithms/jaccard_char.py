from .base import AttackMetric
from ...tags import *

class JaccardChar(AttackMetric):

    NAME = "Jaccard Char Similarity"
    TAGS = { * TAG_ALL_LANGUAGE }

    def calc_score(self, senA : str, senB : str) -> float:
        """
        Args:
            senA: First sentence.
            senB: Second sentence.

        Returns:
            Jaccard char similarity of two sentences.
        
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
    