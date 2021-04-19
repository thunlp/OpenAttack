import OpenAttack
import datasets

def dataset_mapping(x):
    return {
        "x": x["review_body"],
        "y": x["stars"],
    }

def main():
    print("Loading chinese processor and substitute")
    chinese_processor = OpenAttack.text_processors.ChineseTextProcessor()
    chinese_substitute = OpenAttack.substitutes.ChineseHowNetSubstitute()

    print("New Attacker")
    attacker = OpenAttack.attackers.PWWSAttacker(processor=chinese_processor, substitute=chinese_substitute)

    print("Building model")
    clsf = OpenAttack.loadVictim("BERT.AMAZON_ZH").to("cuda:0")

    print("Loading dataset")
    dataset = datasets.load_dataset("amazon_reviews_multi",'zh',split="train[:5]").map(function=dataset_mapping)

    print("Start attack")
    attack_eval = OpenAttack.attack_evals.ChineseAttackEval(attacker, clsf)
    attack_eval.eval(dataset, visualize=True)

if __name__ == "__main__":
    main()
