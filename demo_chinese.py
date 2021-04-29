import OpenAttack
import datasets

def dataset_mapping(x):
    return {
        "x": x["review_body"],
        "y": x["stars"],
    }

import multiprocessing
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
    
def main():
    print("Loading chinese processor and substitute")
    chinese_processor = OpenAttack.text_processors.ChineseTextProcessor()
    chinese_substitute = OpenAttack.substitutes.ChineseCiLinSubstitute()

    print("New Attacker")
    attacker = OpenAttack.attackers.PWWSAttacker(processor=chinese_processor, substitute=chinese_substitute, threshold=None)

    print("Building model")
    clsf = OpenAttack.loadVictim("BERT.AMAZON_ZH").to("cuda:0")

    print("Loading dataset")
    dataset = datasets.load_dataset("amazon_reviews_multi",'zh',split="train[:20]").map(function=dataset_mapping)

    print("Start attack")
    options = {
        "success_rate": True,
        "fluency": True,
        "mistake": True,
        "semantic": True,
        "levenstein": True,
        "word_distance": True,
        "modification_rate": True,
        "running_time": True,
    }
    attack_eval = OpenAttack.attack_evals.ChineseAttackEval(attacker, clsf, **options, num_process=2)
    attack_eval.eval(dataset, visualize=True, progress_bar=True)

if __name__ == "__main__":
    main()
