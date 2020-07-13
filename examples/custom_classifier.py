import TAADToolbox as tat
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer

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

def main():
    dataset = tat.DataManager.load("Dataset.SST.sample")[:10]

    clsf = MyClassifier()
    attacker = tat.attackers.GeneticAttacker()
    attack_eval = tat.attack_evals.DefaultAttackEval(attacker, clsf)
    print( attack_eval.eval(dataset) )

if __name__ == "__main__":
    main()