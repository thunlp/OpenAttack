import OpenAttack
import nltk
import numpy as np
from tqdm import tqdm

def dataset_mapping(x):
    return {
        "x": x["sentence"],
        "y": 1 if x["label"] > 0.5 else 0,
    }
    
def main():
    OpenAttack.DataManager.download("Dataset.SST")
    OpenAttack.DataManager.download("Victim.ROBERTA.SST")

    print("New Attacker")
    attacker = OpenAttack.attackers.PWWSAttacker()

    print("Build model")
    clsf = OpenAttack.DataManager.loadVictim("ROBERTA.SST")

    dataset = datasets.load_dataset("sst", split="train[:20]").map(function=dataset_mapping)

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

if __name__ == "__main__":
    main()
