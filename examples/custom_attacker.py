'''
This example code shows how to design a simple customized attack model which shuffles the tokens in the original sentence.
'''
import OpenAttack
import random

class MyAttacker(OpenAttack.Attacker):
    def __init__(self, processor = OpenAttack.DefaultTextProcessor()):
        self.processor = processor
        # We add parameter ``processor`` to specify the :py:class:`.TextProcessor` which is used for tokenization and detokenization.
        # By default, :py:class:`.DefaultTextProcessor` is used. 
    
    def __call__(self, clsf, x_orig, target=None):
        # Generate a potential adversarial example
        x_new = self.swap(x_orig)
        
        # Get the preidictions of victim classifier
        y_orig, y_new = clsf.get_pred([ x_orig, x_new ])

        # Check for untargeted or targeted attack
        if (target is None and y_orig != y_new) or target == y_new:
            return x_new, y_new
        else:
            # Failed
            return None
    
    def swap(self, sentence):
        # Get tokens of sentence
        tokens = [ token for token, pos in self.processor.get_tokens(sentence) ]
        
        # Shuffle tokens to generate a potential adversarial example
        random.shuffle(tokens)
        
        # Return the potential adversarial example
        return self.processor.detokenizer(tokens)

def main():
    clsf = OpenAttack.DataManager.load("Victim.BiLSTM.SST")
    dataset = OpenAttack.DataManager.load("Dataset.SST.sample")[:10]

    attacker = MyAttacker()
    attack_eval = OpenAttack.attack_evals.DefaultAttackEval(attacker, clsf)
    attack_eval.eval(dataset, visualize=True)

if __name__ == "__main__":
    main()