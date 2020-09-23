import OpenAttack
def get_attackers(dataset, clsf):

    rules = OpenAttack.attackers.SEAAttacker.get_rules(clsf, dataset)
    triggers = OpenAttack.attackers.UATAttacker.get_triggers(clsf, dataset, word2id=clsf.word2id, embedding=clsf.embedding)


    attackers = [
        OpenAttack.attackers.FDAttacker(word2id=clsf.word2id, embedding=clsf.embedding),
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
        OpenAttack.attackers.PSOAttacker(),
    ]
    return attackers