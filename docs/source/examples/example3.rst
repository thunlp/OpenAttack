============================
Customized Attack Model
============================

Attacker is the core module of OpenAttack. In this example, we write a new attacker which swaps tokens
randomly. This is a simple way to generate adversarial samples and it requires a capacity of "Blind".


Initialize Attacker with Options
----------------------------------

.. code-block:: python
    :linenos:
    
    from OpenAttack.tags import Tag
    from OpenAttack.text_process.tokenizer import PunctTokenizer
    class MyAttacker(OpenAttack.attackers.ClassificationAttacker):
        TAGS = { Tag("english", "lang"), Tag("get_pred", "victim") }
        def __init__(self):
            self.tokenizer = PunctTokenizer()

We create a new class called ``MyAttacker`` and create a ``PunctTokenizer`` in its initialization phase of ``MyAttacker`` for tokenization and detokenization.

Besides writing the __init__ method, we also indicate the attacker's supported language and required capabilities via the ``TAGS`` attribute.

The ``TAGS`` are used to help ``OpenAttack`` automatically check the parameters to avoid situations where attacker and victim are using different languages or victim model has insufficient capabilities.


Randomly Swap Tokens
----------------------------------------

.. code-block:: python
    :linenos:

    def swap(self, tokens):
        random.shuffle(tokens)
        return tokens


In ``swap`` method, we shuffle the ``tokens`` of input sentence to generate a candidate.

Check Candidate Sentence and Return
-------------------------------------------

.. code-block:: python
    :linenos:

    def attack(self, victim, input_, goal):
        x_new = self.tokenizer.detokenize(
            self.swap( self.tokenizer.tokenize(input_, pos_tagging=False) )
        )
        y_new = victim.get_pred([ x_new ])
        if goal.check(x_new, y_new):
            return x_new
        return None

``attack`` method is the main procedure of :py:class:`.Attacker`. In this method, we generate a candidate sentence
and use ``Classifier.get_pred`` to get the prediction of victim classifier. Then we check the prediction, return 
``adversarial_sample`` if succeed and return ``None`` if failed.

See :py:class:`.Attacker` for detail.

Complete Code
------------------------------

.. code-block:: python
    :linenos:
    :name: examples/custom_attacker.py
    
    import OpenAttack
    import random
    import datasets

    from OpenAttack.tags import Tag
    from OpenAttack.text_process.tokenizer import PunctTokenizer

    class MyAttacker(OpenAttack.attackers.ClassificationAttacker):
        TAGS = { Tag("english", "lang"), Tag("get_pred", "victim") }
        def __init__(self):
            self.tokenizer = PunctTokenizer()
        
        def attack(self, victim, input_, goal):
            x_new = self.tokenizer.detokenize(
                self.swap( self.tokenizer.tokenize(input_, pos_tagging=False) )
            )
            y_new = victim.get_pred([ x_new ])
            if goal.check(x_new, y_new):
                return x_new
            return None
        
        def swap(self, sentence):
            random.shuffle(sentence)
            return sentence


    def main():
        victim = OpenAttack.loadVictim("BERT.SST")
        def dataset_mapping(x):
            return {
                "x": x["sentence"],
                "y": 1 if x["label"] > 0.5 else 0,
            }
        dataset = datasets.load_dataset("sst").map(function=dataset_mapping)

        attacker = MyAttacker()
        attack_eval = OpenAttack.attack_evals.DefaultAttackEval(attacker, victim)
        attack_eval.eval(dataset, visualize=True)


Run ``python examples/custom_attacker.py`` to see visualized results.