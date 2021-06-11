import sys, os, datasets, time
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".."
))

import OpenAttack
from attackers import get_attackers

def dataset_mapping(x):
    return {
        "x": x["sentence"],
        "y": 1 if x["label"] > 0.5 else 0,
    }

def main():
    import multiprocessing
    if multiprocessing.get_start_method() != "spawn":
        multiprocessing.set_start_method("spawn", force=True)
    dataset = datasets.load_dataset("sst", split="train[:100]").map(function=dataset_mapping)
    clsf = OpenAttack.loadVictim("BERT.SST") # .to("cuda:0")

    attackers = get_attackers(dataset, clsf)

    for attacker in attackers:
        print(attacker.__class__.__name__)
        try:
            print(
                OpenAttack.AttackEval(attacker, clsf).eval(dataset, progress_bar=True),
            )
        except Exception as e:
            raise e
            print(e)
            print("\n")

if __name__ == "__main__":
    main()
