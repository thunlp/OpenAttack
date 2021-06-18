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
    clsf = OpenAttack.loadVictim("BERT.AMAZON_ZH")

    print("Loading dataset")
    dataset = datasets.load_dataset("amazon_reviews_multi",'zh',split="train[:20]").map(function=dataset_mapping)

    print("Start attack")
    attack_eval = OpenAttack.AttackEval(attacker, clsf, metrics=[
        OpenAttack.metric.Fluency(),
        OpenAttack.metric.GrammaticalErrors(),
        OpenAttack.metric.EditDistance(),
        OpenAttack.metric.ModificationRate()
    ])
    attack_eval.eval(dataset, visualize=True, progress_bar=True)

if __name__ == "__main__":
    main()
