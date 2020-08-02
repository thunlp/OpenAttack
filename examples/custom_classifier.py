'''
This example code shows how to use the genetic algorithm-based attack model to attack a customized sentiment analysis model.
'''
import OpenAttack
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# configure access interface of the customized victim model by extending OpenAttack.Classifier.
class MyClassifier(OpenAttack.Classifier):
    def __init__(self):
        # nltk.sentiment.vader.SentimentIntensityAnalyzer is a traditional sentiment classification model.
        self.model = SentimentIntensityAnalyzer()
    
    # access to the classification probability scores with respect input sentences
    def get_prob(self, input_):
        ret = []
        for sent in input_:
            # SentimentIntensityAnalyzer calculates scores of “neg” and “pos” for each instance
            res = self.model.polarity_scores(sent)

            # we use 𝑠𝑜𝑐𝑟𝑒_𝑝𝑜𝑠 / (𝑠𝑐𝑜𝑟𝑒_𝑛𝑒𝑔 + 𝑠𝑐𝑜𝑟𝑒_𝑝𝑜𝑠) to represent the probability of positive sentiment
            # Adding 10^−6 is a trick to avoid dividing by zero.
            prob = (res["pos"] + 1e-6) / (res["neg"] + res["pos"] + 2e-6)

            ret.append(np.array([1 - prob, prob]))
        
        # The get_prob method finally returns a np.ndarray of shape (len(input_), 2). See Classifier for detail.
        return np.array(ret)
        
def main():
    # load Dataset.SST.sample for evaluation
    dataset = OpenAttack.load("Dataset.SST.sample")[:10]
    # choose the costomized classifier as the victim classification model
    clsf = MyClassifier()
    # choose PWWS as the attacker and initialize it with default parameters
    attacker = OpenAttack.attackers.PWWSAttacker()
    # prepare for attacking
    attack_eval = OpenAttack.attack_evals.DefaultAttackEval(attacker, clsf)
    # launch attacks and print attack results 
    attack_eval.eval(dataset, visualize=True)

if __name__ == "__main__":
    main()