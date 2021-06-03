from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
class BLEU:
    def __call__(self, tokenA, tokenB):
        smooth = SmoothingFunction()
        """
            :param list tokenA: The first list of tokens.
            :param list tokenB: The second list of tokens.

            Make sure two list have the same length.
        """
        ref=[tokenA]
        cand=tokenB
        return sentence_bleu(ref,cand,smoothing_function=smooth.method1)