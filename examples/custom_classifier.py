import OpenAttack
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Firstly, a new classifier is implemented by extending OpenAttack.Classifier.
class MyClassifier(OpenAttack.Classifier):
    def __init__(self):
        self.model = SentimentIntensityAnalyzer()
        # nltk.sentiment.vader.SentimentIntensityAnalyzer is a tradictional sentiment classification model.

    def get_prob(self, input_):
        ret = []
        for sent in input_:
            res = self.model.polarity_scores(sent)
            # SentimentIntensityAnalyzer calculates scores of “neg” and “pos” for each instance

            prob = (res["pos"] + 1e-6) / (res["neg"] + res["pos"] + 2e-6)
            # we use 𝑠𝑜𝑐𝑟𝑒_𝑝𝑜𝑠 / (𝑠𝑐𝑜𝑟𝑒_𝑛𝑒𝑔 + 𝑠𝑐𝑜𝑟𝑒_𝑝𝑜𝑠) to represent the probability of positive sentiment
            # Adding 10^−6 is a trick to avoid dividing by zero.

            ret.append(np.array([1 - prob, prob]))
        
        # The get_prob method finally returns a np.ndarray of shape (len(input_), 2). See Classifier for detail.
        return np.array(ret)
        
def main():
    # Secondly, we load Dataset.SST.sample for evaluation and initialize MyClassifier which is defined in the first step. 
    dataset = OpenAttack.load("Dataset.SST.sample")[:10]

    clsf = MyClassifier()
    attacker = OpenAttack.attackers.PWWSAttacker()
    attack_eval = OpenAttack.attack_evals.DefaultAttackEval(attacker, clsf)
    attack_eval.eval(dataset, visualize=True)

if __name__ == "__main__":
    main()