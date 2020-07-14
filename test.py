import OpenAttack

clsf = OpenAttack.DataManager.load("Victim.BiLSTM.SST")
sea = OpenAttack.attackers.SEAAttacker()
dataset = OpenAttack.DataManager.load("Dataset.SST.sample")[:10]

sea.get_rules(clsf, dataset)
