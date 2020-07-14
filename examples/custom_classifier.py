import OpenAttack
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class MyClassifier(OpenAttack.Classifier):
    def __init__(self):
        self.model = SentimentIntensityAnalyzer()

    def get_pred(self, input_):
        return self.get_prob(input_).argmax(axis=1)
        
    def get_prob(self, input_):
        ret = []
        for sent in input_:
            res = self.model.polarity_scores(sent)
            prob = (res["pos"] + 1e-6) / (res["neg"] + res["pos"] + 2e-6)
            ret.append(np.array([1 - prob, prob]))
        return np.array(ret)

    def get_grad(self, input_, labels):
        raise OpenAttack.exceptions.ClassifierNotSupportException(self)

def main():
    dataset = OpenAttack.load("Dataset.SST.sample")[:10]

    clsf = MyClassifier()
    attacker = OpenAttack.attackers.PWWSAttacker()
    attack_eval = OpenAttack.attack_evals.DefaultAttackEval(attacker, clsf)
    attack_eval.eval(dataset, visualize=True)

if __name__ == "__main__":
    main()