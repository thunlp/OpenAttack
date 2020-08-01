'''
This example code shows how to  how to use a genetic algorithm-based attack model to attack BiLSTM on the SST dataset.
'''
import OpenAttack

def main():
    clsf = OpenAttack.DataManager.load("Victim.BiLSTM.SST")
    # Victim.BiLSTM.SST is a pytorch model which is trained on Dataset.SST. It uses Glove vectors for word representation.
    # The load operation returns a PytorchClassifier that can be further used for Attacker and AttackEval.

    dataset = OpenAttack.DataManager.load("Dataset.SST.sample")[:10]
    # Dataset.SST.sample is a list of 1k sentences sampled from test dataset of Dataset.SST.

    attacker = OpenAttack.attackers.GeneticAttacker()
    # After this step, we’ve initialized a GeneticAttacker and uses the default configuration during attack process.

    attack_eval = OpenAttack.attack_evals.DefaultAttackEval(attacker, clsf)
    # DefaultAttackEval is the default implementation for AttackEval which supports seven basic metrics.

    attack_eval.eval(dataset, visualize=True)
    # Using visualize=True in attack_eval.eval can make it displays a visualized result. This function is really useful for analyzing small datasets.

if __name__ == "__main__":
    main()