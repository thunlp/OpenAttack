from ..attacker import Attacker
from ..substitutes import DcesSubstitute
from ..substitutes import EcesSubstitute
import numpy as np
import random


DEFAULT_CONFIG = {

}


class ViperAttacker(Attacker):
    def __init__(self, **kwargs):
        self.config = DEFAULT_CONFIG.copy()
        self.config.update(kwargs)
        self.dces = DcesSubstitute()
        self.eces = EcesSubstitute()
        self.mydict = {}
        self.topn = 12

    def __call__(self, clsf, x_orig, prob):
        """
        * **clsf** : **Classifier** .
        * **x_orig** : Input sentence.
        """
        a = x_orig.rstrip("\n")
        out = []
        for c in a:
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
            if r < prob and len(similar_chars):
                s = np.random.choice(similar_chars, 1, replace=True, p=probs)[0]
            else:
                s = c
            out.append(s)
        ans = "".join(out)
        return ans, clsf.get_pred([ans])[0]
