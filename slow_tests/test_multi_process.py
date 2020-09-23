import OpenAttack
from attackers import get_attackers

def main():
    import multiprocessing
    if multiprocessing.get_start_method() != "spawn":
        multiprocessing.set_start_method("spawn", force=True)
    dataset = OpenAttack.loadDataset("SST")[0][:5]
    clsf = OpenAttack.loadVictim("BiLSTM.SST").to("cuda:0")

    attackers = get_attackers(dataset, clsf)

    for attacker in attackers:
        print(attacker.__class__.__name__)
        try:
            print(
                OpenAttack.attack_evals.MultiProcessAttackEval(attacker, clsf, num_process=2, progress_bar=False).eval(dataset)
            )
        except Exception as e:
            raise e
            print(e)
            print("\n")

if __name__ == "__main__":
    main()
