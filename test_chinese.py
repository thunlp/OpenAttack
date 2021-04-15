import OpenAttack 
dataset = OpenAttack.loadDataset("AMAZON_ZH")[0][:5]
clsf = OpenAttack.DataManager.load("Victim.BERT.AMAZON_ZH")
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

for attacker in attackers:
    print(attacker.__class__.__name__)
    try:
        # print(
        #     OpenAttack.attack_evals.DefaultAttackEval(attacker, clsf, progress_bar=False).eval(dataset)
        # )
        print(
            OpenAttack.attack_evals.MultiProcessAttackEval(attacker, clsf, progress_bar=False).eval(dataset)
        ) 
    except Exception as e:
        print(e)
        print("\n")
