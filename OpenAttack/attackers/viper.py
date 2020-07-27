from ..attacker import Attacker
from ..substitutes import DCESSubstitute
from ..substitutes import ECESSubstitute
import numpy as np
import random


DEFAULT_CONFIG = {
    "prob": 0.3,
    "topn": 12,
    "generations": 120,
    "eces": True
}


class VIPERAttacker(Attacker):
    def __init__(self, **kwargs):
        """
        :param float prob: The probability of changing a char in a sentence. **Default:** 0.3
        :param int topn: Number of substitutes while using DCES substitute. **Default:** 12
        :param int generations: Maximum number of sentences generated per attack. **Default:** 120
        :param bool eces: Use DCES substitute or ECES substitute. **Default:** True

        :Data Requirements: :py:data:`.AttackAssist.DCES`
        :Package Requirements: * **sklearn**
        :Classifier Capacity: Blind

        Text Processing Like Humans Do: Visually Attacking and Shielding NLP Systems. Steffen Eger, Gözde Gül ¸Sahin, Andreas Rücklé, Ji-Ung Lee, Claudia Schulz, Mohsen Mesgar, Krishnkant Swarnkar, Edwin Simpson, Iryna Gurevych. NAACL-HLT 2019.
        `[pdf] <https://www.aclweb.org/anthology/N19-1165>`__
        `[code] <https://github.com/UKPLab/naacl2019-like-humans-visual-attacks>`__
        """
        self.config = DEFAULT_CONFIG.copy()
        self.config.update(kwargs)
        self.dces = DCESSubstitute()
        self.eces = ECESSubstitute()
        self.mydict = {}
        self.topn = self.config["topn"]
        self.prob = self.config["prob"]
        self.generations = self.config["generations"]

    def __call__(self, clsf, x_orig, target=None):
        """
        * **clsf** : **Classifier** .
        * **x_orig** : Input sentence.
        """
        a = x_orig.rstrip("\n")
        y_orig = clsf.get_pred([a])[0]
        for i in range(self.generations):
            out = []
            for c in a:
                if self.config["eces"] is False:
                    if c not in self.mydict:
                        similar_chars, probs = [], []
                        dces_list = self.dces.__call__(c, self.topn)
                        for sc, pr in dces_list:
                            similar_chars.append(sc)
                            probs.append(pr)
                        probs = probs[:len(similar_chars)]
                        probs = probs / np.sum(probs)
                        self.mydict[c] = (similar_chars, probs)
                    else:
                        similar_chars, probs = self.mydict[c]

                    r = random.random()
                    if r < self.prob and len(similar_chars):
                        s = np.random.choice(similar_chars, 1, replace=True, p=probs)[0]
                    else:
                        s = c
                    out.append(s)
                else:
                    r = random.random()
                    if r < self.prob:
                        s = self.eces(c)[0][0]
                    else:
                        s = c
                    out.append(s)
            ans = "".join(out)
            if target is None:
                if clsf.get_pred([ans])[0] != y_orig:
                    return ans, clsf.get_pred([ans])[0]
            else:
                if int(clsf.get_pred([ans])[0]) is int(target):
                    return ans, clsf.get_pred([ans])[0]
        return None