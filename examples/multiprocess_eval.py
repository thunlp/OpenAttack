'''
This example code shows how to using multiprocessing to accelerate adversarial attacks
'''
import OpenAttack
import datasets

def dataset_mapping(x):
    return {
        "x": x["sentence"],
        "y": 1 if x["label"] > 0.5 else 0,
    }
    
    
def main():
    victim = OpenAttack.loadVictim("BERT.SST")
    # Victim.BiLSTM.SST is a pytorch model which is trained on Dataset.SST. It uses Glove vectors for word representation.
    # The load operation returns a PytorchClassifier that can be further used for Attacker and AttackEval.

    dataset = datasets.load_dataset("sst", split="train[:20]").map(function=dataset_mapping)
    # Dataset.SST.sample is a list of 1k sentences sampled from test dataset of Dataset.SST.

    attacker = OpenAttack.attackers.GeneticAttacker()
    # After this step, we’ve initialized a GeneticAttacker and uses the default configuration during attack process.

    attack_eval = OpenAttack.AttackEval(attacker, victim)
    # DefaultAttackEval is the default implementation for AttackEval which supports seven basic metrics.

    attack_eval.eval(dataset, visualize=True, num_workers=4)
    # Using multiprocessing by specify num_workers

if __name__ == "__main__":
    main()