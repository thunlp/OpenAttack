import OpenAttack
sys.path.insert(0, "..")

from attackers import get_attackers
def main():
    dataset = OpenAttack.loadDataset("SST")[0][:5]
    clsf = OpenAttack.loadVictim("BiLSTM.SST")

    attackers = get_attackers(dataset, clsf)

    for attacker in attackers:
        print(attacker.__class__.__name__)
        try:
            print(
                OpenAttack.attack_evals.DefaultAttackEval(attacker, clsf, progress_bar=False).eval(dataset)
            )
        except Exception as e:
            print(e)
            print("\n")

if __name__ == "__main__":
    main()