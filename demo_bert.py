import OpenAttack
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
from tqdm import tqdm
import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'

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


def main():

    print("New Attacker")
    attacker = OpenAttack.attackers.BERTAttacker()

    print("Build model")
    # clsf = make_model()
    clsf = OpenAttack.DataManager.load("Victim.BiLSTM.SST")

    dataset = OpenAttack.DataManager.loadDataset("SST.sample")[2:9]

    print("Start attack")
    options = {
        "success_rate": True,
        "fluency": False,
        "mistake": False,
        "semantic": False,
        "levenstein": True,
        "word_distance": False,
        "modification_rate": True,
        "running_time": True,

        "invoke_limit": 500,
        "average_invoke": True
    }
    attack_eval = OpenAttack.attack_evals.InvokeLimitedAttackEval(attacker, clsf, **options )
    attack_eval.eval(dataset, visualize=True)
    
    # attack_eval = OpenAttack.attack_evals.DefaultAttackEval(attacker, clsf)

    # attack_eval.eval(dataset, visualize=True)
    


if __name__ == "__main__":
    main()
