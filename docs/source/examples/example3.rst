============================
Customized Attack Model
============================

Attacker is the core module of OpenAttack. In this example, we write a new attacker which swaps tokens
randomly. This is a simple way to generate adversarial samples and it requires a capacity of "Blind".


Initialize Attacker with Options
----------------------------------

.. code-block:: python
    :linenos:
    
    class MyAttacker(OpenAttack.Attacker):
        def __init__(self, processor = OpenAttack.text_processors.DefaultTextProcessor()):
            self.processor = processor
            self.max_iter = max_iter

We add parameter ``processor`` to specify the :py:class:`.TextProcessor` which is used for tokenization and detokenization.
By default, :py:class:`.DefaultTextProcessor` is used. Providing default value for each parameter is a good behavior in OpenAttack.
This makes it easier for users to use, and also makes your new attacker easier to be integrated into OpenAttack.

Randomly Swap Tokens
----------------------------------------

.. code-block:: python
    :linenos:

    def swap(self, sentence):
        tokens = [ token for token, pos in self.processor.get_tokens(sentence) ]
        random.shuffle(tokens)
        return self.processor.detokenizer(tokens)


In ``swap`` method, we shuffle the ``tokens`` of input sentence to generate a candidate.

Check Candidate Sentence and Return
-------------------------------------------

.. code-block:: python
    :linenos:

    def __call__(self, clsf, x_orig, target=None):
        x_new = self.swap(x_orig)
        y_orig, y_new = clsf.get_pred([ x_orig, x_new ])

        if (target is None and y_orig != y_new) or target == y_new:
            return x_new, y_new
        else:
            return None

``__call__`` method is the main procedure of :py:class:`.Attacker`. In this method, we generate a candidate sentence
and use ``Classifier.get_pred`` to get the prediction of victim classifier. Then we check the prediction, return 
``(adversarial_sample, adversarial_prediction)`` if succeed and return ``None`` if failed.

See :py:class:`.Attacker` for detail.

Complete Code
------------------------------

.. code-block:: python
    :linenos:
    :name: examples/custom_attacker.py
    
    import OpenAttack
    import random

    class MyAttacker(OpenAttack.Attacker):
        def __init__(self, processor = OpenAttack.text_processors.DefaultTextProcessor()):
            self.processor = processor
        
        def __call__(self, clsf, x_orig, target=None):
            x_new = self.swap(x_orig)
            y_orig, y_new = clsf.get_pred([ x_orig, x_new ])

            if (target is None and y_orig != y_new) or target == y_new:
                return x_new, y_new
            else:
                return None
        
        def swap(self, sentence):
            tokens = [ token for token, pos in self.processor.get_tokens(sentence) ]
            random.shuffle(tokens)
            return self.processor.detokenizer(tokens)

    def main():
        clsf = OpenAttack.DataManager.load("Victim.BiLSTM.SST")
        dataset = OpenAttack.DataManager.load("Dataset.SST.sample")[:10]

        attacker = MyAttacker()
        attack_eval = OpenAttack.attack_evals.DefaultAttackEval(attacker, clsf)
        attack_eval.eval(dataset, visualize=True)

    if __name__ == "__main__":
        main()

Run ``python examples/custom_attacker.py`` to see visualized results.