from OpenAttack import substitute
import sys, os
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".."
))

import OpenAttack
def get_attackers_on_chinese(dataset, clsf):
    chinese_processor = OpenAttack.text_processors.ChineseTextProcessor()
    chinese_substitute = OpenAttack.substitutes.ChineseHowNetSubstitute()
    #Attackers that current support Chinese: SememePSO TextFooler PWWS Genetic FD TextBugger
    attackers = [
        OpenAttack.attackers.FDAttacker(word2id=clsf.word2id, embedding=clsf.embedding, processor=chinese_processor, substitute=chinese_substitute),
        OpenAttack.attackers.TextBuggerAttacker(processor=chinese_processor),
        OpenAttack.attackers.TextFoolerAttacker(processor=chinese_processor, substitute=chinese_substitute),
        OpenAttack.attackers.GeneticAttacker(processor=chinese_processor, substitute=chinese_substitute, skip_words=["的", "了", "着"]),
        OpenAttack.attackers.PWWSAttacker(processor=chinese_processor, substitute=chinese_substitute),
        OpenAttack.attackers.PSOAttacker(processor=chinese_processor, substitute=chinese_substitute)
    ]
    return attackers