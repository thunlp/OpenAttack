import OpenAttack
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
from tqdm import tqdm


def make_model():
    class MyClassifier(OpenAttack.Classifier):
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
            raise OpenAttack.exceptions.ClassifierNotSupportException(self)

    return MyClassifier()


def main():

    print("New Attacker")
    attacker = OpenAttack.attackers.PWWSAttacker()

    print("Build model")
    clsf = make_model()

    dataset = OpenAttack.DataManager.load("Dataset.SST.sample")[:10]

    print("Start attack")
    options = {
        "success_rate": True,   # 成功率
        "fluency": True,       # 流畅度
        "mistake": True,       # 语法错误
        "semantic": True,      # 语义匹配度
        "levenstein": True,    # 编辑距离
        "word_distance": False, # 应用词级别编辑距离

        "invoke_limit": 500,
        "average_invoke": True
    }
    attack_eval = OpenAttack.attack_evals.InvokeLimitedAttackEval(attacker, clsf, progress_bar=True, **options )
    attack_eval.eval(dataset, visualize=True)

if __name__ == "__main__":
    main()
