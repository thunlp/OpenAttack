import sys, os
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".."
))

import OpenAttack
def get_attackers_on_chinese(dataset, clsf):
    chinese_processor = OpenAttack.text_processors.ChineseTextProcessor

    #Attackers that current support Chinese: SememePSO TextFooler PWWS Genetic FD TextBugger
    attackers = [
        OpenAttack.attackers.FDAttacker(word2id=clsf.config["word2id"], embedding=clsf.config["embedding"]),
        OpenAttack.attackers.TextBuggerAttacker(processor=chinese_processor),
        OpenAttack.attackers.TextFoolerAttacker(processor=chinese_processor),
        OpenAttack.attackers.GeneticAttacker(processor=chinese_processor),
        OpenAttack.attackers.PWWSAttacker(processor=chinese_processor),
        OpenAttack.attackers.PSOAttacker(processor=chinese_processor)
    ]
    return attackers