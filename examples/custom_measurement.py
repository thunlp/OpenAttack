import OpenAttack
from nltk.translate.bleu_score import sentence_bleu

class CustomAttackEval(OpenAttack.attack_evals.DefaultAttackEval):
    def __init__(self, attacker, clsf, processor=OpenAttack.DefaultTextProcessor(), **kwargs):
        super().__init__(attacker, clsf, processor=processor, **kwargs)
        self.__processor = processor
        
    def measure(self, x_orig, x_adv):
        info = super().measure(x_orig, x_adv)

        token_orig = [token for token, pos in self.__processor.get_tokens(x_orig)]
        token_adv = [token for token, pos in self.__processor.get_tokens(x_adv)]
        info["Bleu"] = sentence_bleu([x_orig], x_adv)

        return info
    
    def update(self, info):
        info = super().update(info)
        if info["Succeed"]:
            self.__result["bleu"] += info["Bleu"]
        return info

    
    def clear(self):
        super().clear()
        self.__result = { "bleu": 0 }
    
    def get_result(self):
        result = super().get_result()
        result["Avg. Bleu"] = self.__result["bleu"] / result["Successful Instances"]
        return result

def main():
    clsf = OpenAttack.load("Victim.BiLSTM.SST")
    dataset = OpenAttack.load("Dataset.SST.sample")[:10]

    attacker = OpenAttack.attackers.PWWSAttacker()
    attack_eval = CustomAttackEval(attacker, clsf)
    attack_eval.eval(dataset, visualize=True)

if __name__ == "__main__":
    main()