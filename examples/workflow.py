import OpenAttack

def main():
    word_vector = OpenAttack.DataManager.load("Glove")
    model = OpenAttack.DataManager.load("Victim.BiLSTM.SST")
    dataset = OpenAttack.DataManager.load("Dataset.SST.sample")[:10]

    clsf = OpenAttack.classifiers.PytorchClassifier(model, 
                word2id=word_vector.word2id, embedding=word_vector.get_vecmatrix(), 
                token_unk= "UNK", require_length=True, device="cpu")
    attacker = OpenAttack.attackers.GeneticAttacker()
    attack_eval = OpenAttack.attack_evals.DefaultAttackEval(attacker, clsf)
    print( attack_eval.eval(dataset) )

if __name__ == "__main__":
    main()