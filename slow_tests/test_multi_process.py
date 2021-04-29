import sys, os, datasets, time
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".."
))

import OpenAttack
from attackers import get_attackers
from wrapper import TimeCalcClsf

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
    clsf = OpenAttack.loadVictim("BERT.SST").to("cuda:0")

    attackers = get_attackers(dataset, clsf)

    for attacker in attackers:
        print(attacker.__class__.__name__)
        time_clsf = TimeCalcClsf(clsf)
        try:
            st = time.perf_counter()
            print(
                OpenAttack.attack_evals.DefaultAttackEval(attacker, time_clsf, num_process=2).eval(dataset, progress_bar=True),
                time_clsf.total_time,
                time.perf_counter() - st
            )
        except Exception as e:
            raise e
            print(e)
            print("\n")

if __name__ == "__main__":
    main()
