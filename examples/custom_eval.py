'''
This example code shows how to design a customized attack evaluation metric, namely BLEU score.
'''
import OpenAttack
from nltk.translate.bleu_score import sentence_bleu

class CustomAttackEval(OpenAttack.DefaultAttackEval):
    def __init__(self, attacker, clsf, processor=OpenAttack.DefaultTextProcessor(), **kwargs):
        super().__init__(attacker, clsf, processor=processor, **kwargs)
        self.__processor = processor
        # We extend :py:class:`.DefaultAttackEval` and use ``processor`` option to specify 
        # the :py:class:`.TextProcessor` used in our ``CustomAttackEval``.
    
    def measure(self, x_orig, x_adv):
        # Invoke the original ``measure`` method to get measurements
        info = super().measure(x_orig, x_adv)
        if info["Succeed"]:
            # Add ``Blue`` score which is calculated by **NLTK toolkit** if attack succeed.
            token_orig = [token for token, pos in self.__processor.get_tokens(x_orig)]
            token_adv = [token for token, pos in self.__processor.get_tokens(x_adv)]
            info["Bleu"] = sentence_bleu([x_orig], x_adv)
        return info
    
    def update(self, info):
        info = super().update(info)
        
        if info["Succeed"]:
            # Add bleu score that we just calculated to the total result.
            self.__result["bleu"] += info["Bleu"]
        return info
    
    def clear(self):
        super().clear()
        self.__result = { "bleu": 0 }
        # Clear results
    
    def get_result(self):
        result = super().get_result()
        
        # Calculate average bleu scores and return.
        result["Avg. Bleu"] = self.__result["bleu"] / result["Successful Instances"]
        return result
    
def main():
    clsf = OpenAttack.load("Victim.BiLSTM.SST")
    dataset = OpenAttack.load("Dataset.SST.sample")[:10]

    attacker = OpenAttack.attackers.GeneticAttacker()
    attack_eval = CustomAttackEval(attacker, clsf)
    attack_eval.eval(dataset, visualize=True)

if __name__ == "__main__":
    main()