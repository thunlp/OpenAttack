import OpenAttack
dataset = OpenAttack.loadDataset("SST")[0][:5]
clsf = OpenAttack.loadVictim("BiLSTM.SST")

rules = OpenAttack.attackers.SEAAttacker.get_rules(clsf, dataset)
triggers = OpenAttack.attackers.UATAttacker.get_triggers(clsf, dataset, word2id=clsf.config["word2id"], embedding=clsf.config["embedding"])


attackers = [
    OpenAttack.attackers.FDAttacker(word2id=clsf.config["word2id"], embedding=clsf.config["embedding"]),
    OpenAttack.attackers.SEAAttacker(rules=rules),
    OpenAttack.attackers.UATAttacker(triggers=triggers),
    OpenAttack.attackers.TextBuggerAttacker(),
    OpenAttack.attackers.TextFoolerAttacker(),
    OpenAttack.attackers.VIPERAttacker(),
    OpenAttack.attackers.DeepWordBugAttacker(),
    OpenAttack.attackers.GANAttacker(),
    OpenAttack.attackers.GeneticAttacker(),
    OpenAttack.attackers.HotFlipAttacker(),
    OpenAttack.attackers.PWWSAttacker(),
    OpenAttack.attackers.SCPNAttacker(),
]

for attacker in attackers:
    OpenAttack.attack_evals.DefaultAttackEval(attacker, clsf, progress_bar=False).eval(dataset)
