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
        "x": x["review_body"],
        "y": x["stars"],
    }

class MultiprocessInvoke(OpenAttack.attack_evals.multi_process.MultiProcessEvalMixin, OpenAttack.attack_evals.ChineseAttackEval):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

import multiprocessing
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
    
def main():
    print("Loading chinese processor and substitute")
    chinese_processor = OpenAttack.text_processors.ChineseTextProcessor()
    chinese_substitute = OpenAttack.substitutes.ChineseWord2VecSubstitute()

    print("New Attacker")
    attacker = OpenAttack.attackers.PWWSAttacker(processor=chinese_processor, substitute=chinese_substitute)

    print("Building model")
    clsf = OpenAttack.loadVictim("BERT.AMAZON_ZH").to("cuda:0")

    print("Loading dataset")
    dataset = datasets.load_dataset("amazon_reviews_multi",'zh',split="train[:5]").map(function=dataset_mapping)

    print("Start attack")
    options = {
        "success_rate": True,
        "fluency": False,
        "mistake": True,
        "semantic": True,
        "levenstein": True,
        "word_distance": True,
        "modification_rate": True,
        "running_time": True,
    }
    attack_eval = MultiprocessInvoke(attacker, clsf, **options, num_process=2, progress_bar=True)
    attack_eval.eval(dataset, visualize=True)

if __name__ == "__main__":
    main()
