import OpenAttack
import random

class MyAttacker(OpenAttack.Attacker):
    def __init__(self, processor = OpenAttack.text_processors.DefaultTextProcessor()):
        self.processor = processor
        # We add parameter ``processor`` to specify the :py:class:`.TextProcessor` which is used for tokenization and detokenization.
        # By default, :py:class:`.DefaultTextProcessor` is used. 
    
    def __call__(self, clsf, x_orig, target=None):
        x_new = self.swap(x_orig)
        # Generate a candidate sentence

        y_orig, y_new = clsf.get_pred([ x_orig, x_new ])
        # Get the preidictions of victim classifier

        if (target is None and y_orig != y_new) or target == y_new:
            # Check for untargeted or targeted attack
            return x_new, y_new
        else:
            # Failed
            return None
    
    def swap(self, sentence):
        tokens = [ token for token, pos in self.processor.get_tokens(sentence) ]
        # Get tokens of sentence

        random.shuffle(tokens)
        # Shuffle tokens to generate a candidate sentence

        return self.processor.detokenizer(tokens)
        # Return the candidate sentence

def main():
    clsf = OpenAttack.DataManager.load("Victim.BiLSTM.SST")
    dataset = OpenAttack.DataManager.load("Dataset.SST.sample")[:10]

    attacker = MyAttacker()
    attack_eval = OpenAttack.attack_evals.DefaultAttackEval(attacker, clsf)
    attack_eval.eval(dataset, visualize=True)

if __name__ == "__main__":
    main()