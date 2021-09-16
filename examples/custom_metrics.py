'''
This example code shows how to design a customized attack evaluation metric.
'''
import OpenAttack
import datasets

class SentenceLength(OpenAttack.AttackMetric): # extend the AttackMetric class
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
    victim = OpenAttack.loadVictim("BERT.SST")
    dataset = datasets.load_dataset("sst", split="train[:20]").map(function=dataset_mapping)

    attacker = OpenAttack.attackers.PWWSAttacker()
    attack_eval = OpenAttack.AttackEval(attacker, victim, metrics=[
        SentenceLength()
    ])
    attack_eval.eval(dataset, visualize=True)

if __name__ == "__main__":
    main()