import TAADToolbox as tat
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np


def make_model():
    class MyClassifier(tat.Classifier):
        def __init__(self):
            self.model = SentimentIntensityAnalyzer()

        def get_pred(self, input_):
            ret = []
            for sent in input_:
                res = self.model.polarity_scores(sent)
                prob = (res["pos"] + 1e-6) / (res["neg"] + res["pos"] + 2e-6)
                ret.append(0 if prob < 0.5 else 1)
            return np.array(ret)

        def get_prob(self, input_):
            ret = []
            for sent in input_:
                res = self.model.polarity_scores(sent)
                prob = (res["pos"] + 1e-6) / (res["neg"] + res["pos"] + 2e-6)
                ret.append(np.array([1 - prob, prob]))
            return np.array(ret)

        def get_grad(self, input_, labels):
            raise tat.exceptions.ClassifierNotSupportException(self)

    return MyClassifier()


def main():
    print("Download SCPN")
    tat.DataManager.download("SCPN")
    print("New Attacker")
    attacker = tat.attackers.SCPNAttacker()

    print("Build model")
    clsf = make_model()

    sentence = "The quick brown fox jumps over a lazy dog.".lower()
    print("Start attack")
    print("Original prediction:")
    print((sentence, clsf.get_pred([sentence])[0]))
    print("Attack result:")
    print(attacker(clsf, sentence))


if __name__ == "__main__":
    main()
