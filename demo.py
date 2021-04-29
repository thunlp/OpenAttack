import OpenAttack
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import datasets

def make_model():
    class MyClassifier(OpenAttack.Classifier):
        def __init__(self):
            try:
                self.model = SentimentIntensityAnalyzer()
            except LookupError:
                nltk.download('vader_lexicon')
                self.model = SentimentIntensityAnalyzer()
            
        def get_prob(self, input_):
            ret = []
            for sent in input_:
                res = self.model.polarity_scores(sent)
                prob = (res["pos"] + 1e-6) / (res["neg"] + res["pos"] + 1e-6)
                ret.append(np.array([1 - prob, prob]))
            return np.array(ret)
    return MyClassifier()

def dataset_mapping(x):
    return {
        "x": x["sentence"],
        "y": 1 if x["label"] > 0.5 else 0,
    }

import multiprocessing
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
    
def main():

    print("New Attacker")
    attacker = OpenAttack.attackers.PWWSAttacker()

    print("Build model")
    clsf = OpenAttack.loadVictim("BERT.SST").to("cuda:0")

    dataset = datasets.load_dataset("sst", split="train[:100]").map(function=dataset_mapping)

    print("Start attack")
    options = {
        "success_rate": True,
        "fluency": True,
        "mistake": True,
        "semantic": True,
        "levenstein": True,
        "word_distance": True,
        "modification_rate": True,
        "running_time": True,

        "invoke_limit": 500,
        "average_invoke": True
    }
    attack_eval = OpenAttack.attack_evals.InvokeLimitedAttackEval(attacker, clsf, **options, num_process=4)
    attack_eval.eval(dataset, visualize=True, progress_bar=True)

if __name__ == "__main__":
    main()
