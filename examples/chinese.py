'''
This example code shows how to conduct adversarial attacks against a Chinese review classification model using PWWS
'''
import OpenAttack
import datasets

def dataset_mapping(x):
    return {
        "x": x["review_body"],
        "y": x["stars"],
    }

def main():
    print("New Attacker")
    attacker = OpenAttack.attackers.PWWSAttacker(lang="chinese")

    print("Building model")
    victim = OpenAttack.loadVictim("BERT.AMAZON_ZH").to("cuda:0")

    print("Loading dataset")
    dataset = datasets.load_dataset("amazon_reviews_multi",'zh',split="train[:20]").map(function=dataset_mapping)

    print("Start attack")
    attack_eval = OpenAttack.AttackEval(attacker, victim)
    attack_eval.eval(dataset, visualize=True, progress_bar=True)

if __name__ == "__main__":
    main()
