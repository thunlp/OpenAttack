'''
This example code shows how to  how to use a genetic algorithm-based attack model to attack BiLSTM on the SST dataset.
'''
import OpenAttack
import datasets

def dataset_mapping(x):
    return {
        "x": x["sentence"],
        "y": 1 if x["label"] > 0.5 else 0,
    }
    
    
def main():
    clsf = OpenAttack.loadVictim("BERT.SST")
    # Victim.BiLSTM.SST is a pytorch model which is trained on Dataset.SST. It uses Glove vectors for word representation.
    # The load operation returns a PytorchClassifier that can be further used for Attacker and AttackEval.

    dataset = datasets.load_dataset("sst", split="train[:20]").map(function=dataset_mapping)
    # We load sst dataset using `datasets` package, and map the fields.

    attacker = OpenAttack.attackers.GeneticAttacker()
    # After this step, we’ve initialized a GeneticAttacker and uses the default configuration during attack process.

    attack_eval = OpenAttack.AttackEval(attacker, clsf)
    # DefaultAttackEval is the default implementation for AttackEval which supports seven basic metrics.

    attack_eval.eval(dataset, visualize=True)
    # Using visualize=True in attack_eval.eval can make it displays a visualized result. This function is really useful for analyzing small datasets.

if __name__ == "__main__":
    main()