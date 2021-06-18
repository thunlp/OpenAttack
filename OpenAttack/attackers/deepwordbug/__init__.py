from ..classification import ClassificationAttacker, Classifier, ClassifierGoal
from ...text_process.tokenizer import Tokenizer, get_default_tokenizer
from ...utils import check_language
from ...tags import TAG_English, Tag
import numpy as np


homos = {
         '-': '˗', '9': '৭', '8': 'Ȣ', '7': '𝟕', '6': 'б', '5': 'Ƽ', '4': 'Ꮞ', '3': 'Ʒ', '2': 'ᒿ', '1': 'l', '0': 'O',
         "'": '`', 'a': 'ɑ', 'b': 'Ь', 'c': 'ϲ', 'd': 'ԁ', 'e': 'е', 'f': '𝚏', 'g': 'ɡ', 'h': 'հ', 'i': 'і', 'j': 'ϳ',
         'k': '𝒌', 'l': 'ⅼ', 'm': 'ｍ', 'n': 'ո', 'o': 'о', 'p': 'р', 'q': 'ԛ', 'r': 'ⲅ', 's': 'ѕ', 't': '𝚝', 'u': 'ս',
         'v': 'ѵ', 'w': 'ԝ', 'x': '×', 'y': 'у', 'z': 'ᴢ'
}


class DeepWordBugAttacker(ClassificationAttacker):
    @property
    def TAGS(self):
        return { self.__lang_tag, Tag("get_pred", "victim"), Tag("get_prob", "victim") }

    def __init__(self, 
            token_unk = "<UNK>",
            scoring = "replaceone",
            transform = "homoglyph",
            power = 5,
            tokenizer : Tokenizer = None
        ):
        """
        Black-box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers. Ji Gao, Jack Lanchantin, Mary Lou Soffa, Yanjun Qi. IEEE SPW 2018.
        `[pdf] <https://ieeexplore.ieee.org/document/8424632>`__
        `[code] <https://github.com/QData/deepWordBug>`__

        Args:
            token_unk: The token id or the token name for out-of-vocabulary words in victim model. **Default:** ``"<UNK>"``
            scoring: Scoring function used to compute word importance, must be one of the following: ``["replaceone", "temporal", "tail", "combined"]``. **Default:** replaceone
            transform: Transform function to modify a word, must be one of the following:  ``["homoglyph", "swap"]``. **Default:** homoglyph
            power: Max words to replace. **Default:** 5
            tokenizer: A tokenizer that will be used during the attack procedure. Must be an instance of :py:class:`.Tokenizer`

        :Classifier Capacity:
            * get_pred
            * get_prob

        
        """

        self.token_unk = token_unk
        self.scoring = scoring
        self.transformer = transform
        self.power = power
        
        if tokenizer is None:
            self.tokenizer = get_default_tokenizer(None)
        else:
            self.tokenizer = tokenizer
        self.__lang_tag = TAG_English
        check_language([self.tokenizer], self.__lang_tag)
        
    def attack(self, victim: Classifier, x_orig, goal: ClassifierGoal):
        """
        * **clsf** : **Classifier** .
        * **x_orig** : Input sentence.
        """
        tokens = self.tokenizer.tokenize(x_orig, pos_tagging=False)
        losses = self.scorefunc(self.scoring, victim, tokens, goal)  # 每个词消失后的loss向量
        indices = np.argsort(losses)

        advinputs = tokens[:]
        t = 0
        j = 0
        while j < self.power and t < len(tokens):
            if advinputs[indices[t]] != '' and advinputs[indices[t]] != ' ':
                advinputs[indices[t]] = self.transform(self.transformer, advinputs[indices[t]])
                j += 1
            t += 1

        ret = self.tokenizer.detokenize(advinputs)
        output2 = victim.get_pred([ret])[0]
        if goal.check(ret, output2):
            return ret
        return None

    def scorefunc(self, type_, victim, tokens, goal):
        if type_ == "replaceone":
            return self.replaceone(victim, tokens, goal)
        elif type_ == "temporal":
            return self.temporal(victim, tokens, goal)
        elif type_ == "tail":
            return self.temporaltail(victim, tokens, goal)
        elif type_ == "combined":
            return self.combined(victim, tokens, goal)
        else:
            raise ValueError("Unknown score function %s, %s expected" % (type_, ["replaceone", "temporal", "tail", "combined"]))

    def transform(self, type_, word):
        if type_ == "homoglyph":
            return self.homoglyph(word)
        elif type_ == "swap":
            return self.temporal(word)
        else:
            raise ValueError("Unknown transform function %s, %s expected" % (type_, ["homoglyph", "swap"]))

    # scoring functions
    def replaceone(self, victim, tokens, goal):
        losses = np.zeros(len(tokens))
        for i in range(len(tokens)):
            tempinputs = tokens[:]  # ##
            tempinputs[i] = self.token_unk
            tempoutput = victim.get_prob([ self.tokenizer.detokenize(tempinputs) ])
            if not goal.targeted:
                losses[i] = tempoutput[0][goal.target]
            else:
                losses[i] = 1 - tempoutput[0][goal.target]
        return losses

    def temporal(self, victim, tokens, goal):
        losses1 = np.zeros(len(tokens))
        dloss = np.zeros(len(tokens))
        for i in range(len(tokens)):
            tempinputs = tokens[: i + 1]
            tempoutput = victim.get_prob([self.tokenizer.detokenize(tempinputs)])
            if not goal.targeted:
                losses1[i] = tempoutput[0][goal.target]
            else:
                losses1[i] = 1 - tempoutput[0][goal.target]
        for i in range(1, len(tokens)):
            dloss[i] = abs(losses1[i] - losses1[i - 1])
        return dloss

    def temporaltail(self, victim, tokens, goal):
        losses1 = np.zeros(len(tokens))
        dloss = np.zeros(len(tokens))
        for i in range(len(tokens)):
            tempinputs = tokens[i:]
            tempoutput = victim.get_prob([self.tokenizer.detokenize(tempinputs)])
            if not goal.targeted:
                losses1[i] = tempoutput[0][goal.target]
            else:
                losses1[i] = 1 - tempoutput[0][goal.target]
        for i in range(1, len(tokens)):
            dloss[i] = abs(losses1[i] - losses1[i - 1])
        return dloss

    def combined(self, victim, tokens, goal):
        temp = self.temporal(victim, tokens, goal)
        temptail = self.temporaltail(victim, tokens, goal)
        return (temp+temptail) / 2

    # transform functions
    def homoglyph(self, word):
        s = np.random.randint(0, len(word))
        if word[s] in homos:
            rletter = homos[word[s]]
        else:
            rletter = word[s]
        cword = word[:s] + rletter + word[s+1:]
        return cword

    def swap(self, word):
        if len(word) != 1:
            s = np.random.randint(0, len(word)-1)
            cword = word[:s] + word[s+1] + word[s] + word[s+2:]
        else:
            cword = word
        return cword
