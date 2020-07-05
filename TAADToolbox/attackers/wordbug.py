from ..attacker import Attacker
import numpy as np


DEFAULT_CONFIG = {
    "unk": "unk",  # unk token
    "scoring": "replaceone",  # replaceone, temporal, tail, combined
    "transformer": "homoglyph",  # homoglyph, swap
    "power": 5
}
homos = {
         '-': '˗', '9': '৭', '8': 'Ȣ', '7': '𝟕', '6': 'б', '5': 'Ƽ', '4': 'Ꮞ', '3': 'Ʒ', '2': 'ᒿ', '1': 'l', '0': 'O',
         "'": '`', 'a': 'ɑ', 'b': 'Ь', 'c': 'ϲ', 'd': 'ԁ', 'e': 'е', 'f': '𝚏', 'g': 'ɡ', 'h': 'հ', 'i': 'і', 'j': 'ϳ',
         'k': '𝒌', 'l': 'ⅼ', 'm': 'ｍ', 'n': 'ո', 'o': 'о', 'p': 'р', 'q': 'ԛ', 'r': 'ⲅ', 's': 'ѕ', 't': '𝚝', 'u': 'ս',
         'v': 'ѵ', 'w': 'ԝ', 'x': '×', 'y': 'у', 'z': 'ᴢ'
}


class WordBugAttacker(Attacker):
    def __init__(self, **kwargs):
        self.config = DEFAULT_CONFIG.copy()
        self.config.update(kwargs)
        self.scoring = self.config["scoring"]
        self.transformer = self.config["transformer"]
        self.power = self.config["power"]

    def __call__(self, clsf, x_orig, target=None):
        import torch
        """
        * **clsf** : **Classifier** .
        * **x_orig** : Input sentence.
        """
        if target is None:
            target = clsf.get_pred([x_orig])[0]
        inputs = x_orig.strip().lower().split(" ")
        losses = self.scorefunc(self.scoring, clsf, inputs, target)  # 每个词消失后的loss向量
        sorted, indices = torch.sort(losses, descending=True)

        advinputs = inputs[:]
        t = 0
        j = 0
        while j < self.power and t < len(inputs):
            if advinputs[indices[t]] != '' and advinputs[indices[t]] != ' ':
                advinputs[indices[t]] = self.transform(self.transformer, advinputs[indices[t]])
                j += 1
            t += 1

        output2 = clsf.get_pred([" ".join(advinputs)])[0]
        return " ".join(advinputs), output2

    def scorefunc(self, type, clsf, inputs, target):
        if "replaceone" in type:
            return self.replaceone(clsf, inputs, target)
        elif "temporal" in type:
            return self.temporal(clsf, inputs, target)
        elif "tail" in type:
            return self.temporaltail(clsf, inputs, target)
        elif "combined" in type:
            return self.combined(clsf, inputs, target)
        else:
            print("error, No scoring func found")

    def transform(self, type, word):
        if "homoglyph" in type:
            return self.homoglyph(word)
        elif "swap" in type:
            return self.temporal(word)
        else:
            print("error, No transform func found")

    # scoring functions
    def replaceone(self, clsf, inputs, target):
        import torch

        losses = torch.zeros(len(inputs))
        for i in range(len(inputs)):
            tempinputs = inputs[:]  # ##
            tempinputs[i] = self.config['unk']
            with torch.no_grad():
                tempoutput = torch.from_numpy(clsf.get_prob([" ".join(tempinputs)]))  # ##
            softmax = torch.nn.Softmax(dim=1)
            nll_lossed = -1 * torch.log(softmax(tempoutput))[0][target].item()
            # losses[i] = F.nll_loss(tempoutput, torch.tensor([[target]], dtype=torch.long), reduce=False)
            losses[i] = nll_lossed  # ##
            # print(" ".join(tempinputs), nll_lossed)
        return losses

    def temporal(self, clsf, inputs, target):
        import torch
        softmax = torch.nn.Softmax(dim=1)

        losses1 = torch.zeros(len(inputs))
        dloss = torch.zeros(len(inputs))
        for i in range(len(inputs)):
            tempinputs = inputs[: i + 1]
            with torch.no_grad():
                tempoutput = torch.from_numpy(clsf.get_prob([" ".join(tempinputs)]))
            # losses1[i] = F.nll_loss(tempoutput, target, reduce=False)
            losses1[i] = -1 * torch.log(softmax(tempoutput))[0][target].item()
            print(" ".join(tempinputs), losses1[i])
        for i in range(1, len(inputs)):
            dloss[i] = abs(losses1[i] - losses1[i - 1])
        return dloss

    def temporaltail(self, clsf, inputs, target):
        import torch
        softmax = torch.nn.Softmax(dim=1)

        losses1 = torch.zeros(len(inputs))
        dloss = torch.zeros(len(inputs))
        for i in range(len(inputs)):
            tempinputs = inputs[i:]
            with torch.no_grad():
                tempoutput = torch.from_numpy(clsf.get_prob([" ".join(tempinputs)]))
            # losses1[i] = F.nll_loss(tempoutput, target, reduce=False)
            losses1[i] = -1 * torch.log(softmax(tempoutput))[0][target].item()
        for i in range(1, len(inputs)):
            dloss[i] = abs(losses1[i] - losses1[i - 1])
        return dloss

    def combined(self, clsf, inputs, target):
        temp = self.temporal(clsf, inputs, target)
        temptail = self.temporaltail(clsf, inputs, target)
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
