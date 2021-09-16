'''
This example code shows how to design a customized attack model (that shuffles the tokens in the original sentence).
'''
import OpenAttack
import random
import datasets

from OpenAttack.tags import Tag
from OpenAttack.text_process.tokenizer import PunctTokenizer

class MyAttacker(OpenAttack.attackers.ClassificationAttacker):
    @property
    def TAGS(self):
        # returns tags can help OpenAttack to check your parameters automatically
        return { self.lang_tag, Tag("get_pred", "victim") }

    def __init__(self, tokenizer = None):
        if tokenizer is None:
            tokenizer = PunctTokenizer()
        self.tokenizer = tokenizer
        self.lang_tag = OpenAttack.utils.get_language([self.tokenizer])
        # We add parameter ``processor`` to specify the :py:class:`.TextProcessor` which is used for tokenization and detokenization.
        # By default, :py:class:`.DefaultTextProcessor` is used. 
    
    def attack(self, victim, input_, goal):
        # Generate a potential adversarial example
        x_new = self.tokenizer.detokenize(
            self.swap( self.tokenizer.tokenize(input_, pos_tagging=False) )
        )
        
        # Get the preidictions of victim classifier
        y_new = victim.get_pred([ x_new ])

        # Check for attack goal
        if goal.check(x_new, y_new):
            return x_new
        # Failed
        return None
    
    def swap(self, sentence):        
        # Shuffle tokens to generate a potential adversarial example
        random.shuffle(sentence)

        # Return the potential adversarial example
        return sentence

def dataset_mapping(x):
    return {
        "x": x["sentence"],
        "y": 1 if x["label"] > 0.5 else 0,
    }

def main():
    victim = OpenAttack.loadVictim("BERT.SST")
    dataset = datasets.load_dataset("sst", split="train[:10]").map(function=dataset_mapping)

    attacker = MyAttacker()
    attack_eval = OpenAttack.AttackEval(attacker, victim)
    attack_eval.eval(dataset, visualize=True)

if __name__ == "__main__":
    main()