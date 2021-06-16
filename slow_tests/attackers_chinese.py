from OpenAttack import substitute
import sys, os
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".."
))

import OpenAttack
def get_attackers_on_chinese(dataset, clsf):
    
    triggers = OpenAttack.attackers.UATAttacker.get_triggers(clsf, dataset, clsf.tokenizer)

    attackers = [
        OpenAttack.attackers.FDAttacker(token_unk=clsf.token_unk, lang="chinese"),
        OpenAttack.attackers.UATAttacker(triggers=triggers, lang="chinese"),
        OpenAttack.attackers.TextBuggerAttacker(lang="chinese"),
        OpenAttack.attackers.GeneticAttacker(lang="chinese", filter_words=["的", "了", "着"]),
        OpenAttack.attackers.PWWSAttacker(lang="chinese"),
        OpenAttack.attackers.PSOAttacker(lang="chinese")
    ]
    return attackers