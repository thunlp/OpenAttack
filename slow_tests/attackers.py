import sys, os
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".."
))

import OpenAttack
def get_attackers(dataset, clsf):

    # rules = OpenAttack.attackers.SEAAttacker.get_rules(clsf, dataset)
    print(clsf.embedding.shape)
    triggers = OpenAttack.attackers.UATAttacker.get_triggers(clsf, dataset, word2id=clsf.word2id, embedding=clsf.embedding)
    print(triggers)

    attackers = [
        OpenAttack.attackers.FDAttacker(word2id=clsf.word2id, embedding=clsf.embedding, token_unk="[UNK]"),
        # OpenAttack.attackers.SEAAttacker(rules=rules),
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
        OpenAttack.attackers.BAEAttacker(),
        OpenAttack.attackers.BERTAttacker()
    ]

    return attackers
