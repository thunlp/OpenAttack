import sys, os
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".."
))
import OpenAttack


from attackers import get_attackers
def main():
    temp = []
    for i in range(32):
        temp.append(i)
    dataset = OpenAttack.loadDataset("SST")["train"].select(temp)
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