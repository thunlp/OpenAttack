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
    print("Download CounterFittedEmbedding")
    tat.DataManager.download("CounterFit")
    print("New Attacker")
    attacker = tat.attackers.GNLAEAttacker()

    print("Build model")
    clsf = make_model()

    sentence = "It is also seems like it wants to really poke at christianity but then loses that in the end much to my chagrin but leaving an inconsistent feel to the movie . Could have been much worse if excesses were taken in sex and violence, but they try to keep this at a minimal despite some disgusting scenes . My final thought is why would Hooper want to make this movie . It obviously took awhile to actually get distributed , then it has to be advertised gruesomely and with Hooper 's name in the title to hopefully make some money on his name and his gore . It is obvious this did not work .".lower()
    print("Start attack")
    print("Original prediction:")
    print((sentence, clsf.get_pred([sentence])[0]))
    print("Attack result:")
    print(attacker(clsf, sentence))


if __name__ == "__main__":
    main()
