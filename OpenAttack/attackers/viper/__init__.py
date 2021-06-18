import numpy as np
import random


from ..classification import ClassificationAttacker, Classifier, ClassifierGoal
from ...text_process.tokenizer import Tokenizer, get_default_tokenizer
from ...utils import get_language, check_language, language_by_name
from ...tags import Tag
from ...attack_assist.substitute.char import DCESSubstitute, ECESSubstitute

DEFAULT_CONFIG = {
    "prob": 0.3,
    "topn": 12,
    "generations": 120,
    "eces": True
}


class VIPERAttacker(ClassificationAttacker):
    
    @property
    def TAGS(self):
        return { self.__lang_tag, Tag("get_pred", "victim") }

    def __init__(self,
            prob : float = 0.3,
            topn : int = 12,
            generations : int = 120,
            method: str = "eces",
        ):
        """
        Text Processing Like Humans Do: Visually Attacking and Shielding NLP Systems. Steffen Eger, Gözde Gül ¸Sahin, Andreas Rücklé, Ji-Ung Lee, Claudia Schulz, Mohsen Mesgar, Krishnkant Swarnkar, Edwin Simpson, Iryna Gurevych. NAACL-HLT 2019.
        `[pdf] <https://www.aclweb.org/anthology/N19-1165>`__
        `[code] <https://github.com/UKPLab/naacl2019-like-humans-visual-attacks>`__

        Args:
            prob: The probability of changing a char in a sentence. **Default:** 0.3
            topn: Number of substitutes while using DCES substitute. **Default:** 12
            generations: Maximum number of sentences generated per attack. **Default:** 120
            method: The method of this attack. Must be one of the following: ``["eces", "dces"]``. **Default:** eces

        :Classifier Capacity:
            * get_pred
        
        """
        
        self.prob = prob
        self.topn = topn
        self.generations = generations
        self.method = method
        if method == "dces":
            self.substitute = DCESSubstitute()
        elif method == "eces":
            self.substitute = ECESSubstitute()
        else:
            raise ValueError("Unknown method `%s` expect `%s`" % (method, ["dces", "eces"]))
        
        self.__lang_tag = get_language([self.substitute])

        self.sim_dict = {}
        
        

    def attack(self, victim: Classifier, sentence : str, goal: ClassifierGoal):
        for _ in range(self.generations):
            out = []
            for c in sentence:
                if self.method == "dces":
                    if c not in self.sim_dict:
                        similar_chars, probs = [], []
                        dces_list = self.substitute(c)[:self.topn]
                        for sc, pr in dces_list:
                            similar_chars.append(sc)
                            probs.append(pr)
                        probs = probs / np.sum(probs)
                        self.sim_dict[c] = (similar_chars, probs)
                    else:
                        similar_chars, probs = self.sim_dict[c]

                    r = random.random()
                    if r < self.prob and len(similar_chars) > 0:
                        s = np.random.choice(similar_chars, 1, replace=True, p=probs)[0]
                    else:
                        s = c
                    out.append(s)
                else:
                    r = random.random()
                    if r < self.prob:
                        s = self.substitute(c)[0][0]
                    else:
                        s = c
                    out.append(s)
            ans = "".join(out)
            pred = victim.get_pred([ans])[0]

            if goal.check(ans, pred):
                return ans
        return None