import OpenAttack
import nltk
import numpy as np
from tqdm import tqdm


def main():
    OpenAttack.DataManager.download("Dataset.SST")
    OpenAttack.DataManager.download("Victim.ROBERTA.SST")

    print("New Attacker")
    attacker = OpenAttack.attackers.PWWSAttacker()

    print("Build model")
    clsf = OpenAttack.DataManager.loadVictim("ROBERTA.SST")

    dataset = OpenAttack.DataManager.loadDataset("SST")[0][:10]

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
