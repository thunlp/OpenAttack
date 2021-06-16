import sys, os, datasets, time
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".."
))

import OpenAttack
from attackers_chinese import get_attackers_on_chinese

def dataset_mapping(x):
    return {
        "x": x["review_body"],
        "y": x["stars"],
    }

def main():
    import multiprocessing
    if multiprocessing.get_start_method() != "spawn":
        multiprocessing.set_start_method("spawn", force=True)
    dataset = datasets.load_dataset("amazon_reviews_multi",'zh',split="train[:5]").map(dataset_mapping)
    
    clsf = OpenAttack.loadVictim("BERT.AMAZON_ZH").to("cuda:0")
    attackers = get_attackers_on_chinese(dataset, clsf)

    for attacker in attackers:
        print(attacker.__class__.__name__)
        try:
            st = time.perf_counter()
            print(
                OpenAttack.AttackEval(attacker, clsf, language="chinese").eval(dataset, progress_bar=True),
                time.perf_counter() - st
            )
        except Exception as e:
            raise e
            print(e)
            print("\n")

if __name__ == "__main__":
    main()