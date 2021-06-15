from .base import AttackMetric
from ...text_process.tokenizer import Tokenizer
class BLEU(AttackMetric):
    NAME = "BLEU"

    def __init__(self, tokenizer : Tokenizer) -> None:
        from nltk.translate.bleu_score import sentence_bleu
        from nltk.translate.bleu_score import SmoothingFunction
        self.smooth = SmoothingFunction()
        self.sentence_bleu = sentence_bleu
        self.tokenizer = tokenizer

    @property
    def TAGS(self):
        if hasattr(self.tokenizer, "TAGS"):
            return self.tokenizer.TAGS
        else:
            return set()

    def calc_score(self, tokenA, tokenB):
        """
            :param list tokenA: The first list of tokens.
            :param list tokenB: The second list of tokens.

            Make sure two list have the same length.
        """
        ref = [ tokenA ]
        cand = tokenB
        return self.sentence_bleu(ref, cand, smoothing_function = self.smooth.method1)

    def after_attack(self, input, adversarial_sample):
        if adversarial_sample is not None:
            return self.calc_score( self.tokenizer.tokenize(input["x"], pos_tagging=False), self.tokenizer.tokenize(adversarial_sample, pos_tagging=False) )
        else:
            return None
