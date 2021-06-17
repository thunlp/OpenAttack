'''
This example code shows how to design a customized attack evaluation metric, namely BLEU score.
'''
import OpenAttack
from nltk.translate.bleu_score import sentence_bleu
import datasets

class SentenceLength(OpenAttack.AttackMetric):
    # name of the metric
    NAME = "Input Length"

    def after_attack(self, input, adversarial_sample):
        # returns the length of input sentence
        return len(input["x"].split(" "))

def dataset_mapping(x):
    return {
        "x": x["sentence"],
        "y": 1 if x["label"] > 0.5 else 0,
    }
    
def main():
    clsf = OpenAttack.load("Victim.BERT.SST")
    dataset = datasets.load_dataset("sst", split="train[:20]").map(function=dataset_mapping)

    attacker = OpenAttack.attackers.GeneticAttacker()
    attack_eval = OpenAttack.AttackEval(attacker, clsf, metrics=[
        SentenceLength()
    ])
    attack_eval.eval(dataset, visualize=True)

if __name__ == "__main__":
    main()