import OpenAttack

def main():
    clsf = OpenAttack.DataManager.load("Victim.BiLSTM.SST")
    dataset = OpenAttack.DataManager.load("Dataset.SST.sample")[:10]

    attacker = OpenAttack.attackers.GeneticAttacker()
    attack_eval = OpenAttack.attack_evals.DefaultAttackEval(attacker, clsf)
    attack_eval.eval(dataset, visualize=True)

if __name__ == "__main__":
    main()