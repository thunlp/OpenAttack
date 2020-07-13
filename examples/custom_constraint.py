import TAADToolbox as tat

class AttackerEvalConstraint(tat.attacker_evals.DefaultAttackerEval):
    def __init__(self, attacker, clsf, mistake_limit=5, **kwargs):
        self.mistake_limit = mistake_limit
        super().__init__(attacker, clsf, mistake=True, **kwargs)
    
    def measure(self, sentA, sentB):
        info = super().measure(sentA, sentB)
        if info["succeed"] and info["mistake"] >= self.mistake_limit:
            info["succeed"] = False
        return info

def main():
    word_vector = tat.DataManager.load("Glove")
    model = tat.DataManager.load("Victim.BiLSTM.SST")
    dataset = tat.DataManager.load("Dataset.SST.sample")[:10]

    clsf = tat.classifiers.PytorchClassifier(model, 
                word2id=word_vector.word2id, embedding=word_vector.get_vecmatrix(), 
                token_unk= "UNK", require_length=True, device="cpu")
    attacker = tat.attackers.PWWSAttacker()
    attacker_eval = AttackerEvalConstraint(attacker, clsf)
    print( attacker_eval.eval(dataset) )

if __name__ == "__main__":
    main()