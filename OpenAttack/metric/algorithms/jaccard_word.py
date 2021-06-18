from .base import AttackMetric
from ...tags import *
from ...text_process.tokenizer import Tokenizer

class JaccardWord(AttackMetric):
    
    NAME = "Jaccard Word Similarity"

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

    def calc_score(self, sentA : str, sentB : str) -> float:
        """
        Args:
            sentA: First sentence.
            sentB: Second sentence.

        Returns:
            Jaccard word similarity of two sentences.
        
        """
        tokenA = self.tokenizer.tokenize(sentA, pos_tagging=False)
        tokenB = self.tokenizer.tokenize(sentB, pos_tagging=False)

        AS=set()
        BS=set()
        for i in range(len(tokenA)):
            AS.add(tokenA[i])
        for i in range(len(tokenB)):
            BS.add(tokenB[i])

        return len(AS&BS)/len(AS|BS)
    
    def after_attack(self, input, adversarial_sample):
        if adversarial_sample is not None:
            return self.calc_score( input["x"], adversarial_sample )
        return None
