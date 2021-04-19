'''
This example code shows how to  how to use a genetic algorithm-based attack model to attack BiLSTM on the SST dataset.
'''
import OpenAttack
import datasets
import multiprocessing
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

def dataset_mapping(x):
    return {
        "x": x["sentence"],
        "y": 1 if x["label"] > 0.5 else 0,
    }
    
    
def main():
    clsf = OpenAttack.loadVictim("BERT.SST").to("cuda")
    # Victim.BiLSTM.SST is a pytorch model which is trained on Dataset.SST. It uses Glove vectors for word representation.
    # The load operation returns a PytorchClassifier that can be further used for Attacker and AttackEval.

    dataset = datasets.load_dataset("sst", split="train[:20]").map(function=dataset_mapping)
    # Dataset.SST.sample is a list of 1k sentences sampled from test dataset of Dataset.SST.

    attacker = OpenAttack.attackers.PWWSAttacker()
    # After this step, we’ve initialized a PWWSAttacker and uses the default configuration during attack process.

    options = {
        "success_rate": True,
        "fluency": False,
        "mistake": False,
        "semantic": False,
        "levenstein": True,
        "word_distance": False,
        "modification_rate": True,
        "running_time": True,
    }

    attack_eval = OpenAttack.attack_evals.MultiProcessAttackEval(attacker, clsf, num_process=4, **options)
    # DefaultAttackEval is the default implementation for AttackEval which supports seven basic metrics.

    attack_eval.eval(dataset, visualize=True)
    # Using visualize=True in attack_eval.eval can make it displays a visualized result. This function is really useful for analyzing small datasets.

if __name__ == "__main__":
    main()