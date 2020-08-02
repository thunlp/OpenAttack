============================================
Customized Measurement
============================================

In this example, we write a new AttackEval to evaluate attackers and report average bleu score.
This example is a bit harder than the other tree examples because a deeper understanding of OpenAttack is needed.

Initialize AttackEval
-----------------------

.. code-block:: python
    :linenos:

    class CustomAttackEval(OpenAttack.DefaultAttackEval):
        def __init__(self, attacker, clsf, processor=OpenAttack.DefaultTextProcessor(), **kwargs):
            super().__init__(attacker, clsf, processor=processor, **kwargs)
            self.__processor = processor
            self.__result = {}

We extend :py:class:`.DefaultAttackEval` and use ``processor`` option to specify the :py:class:`.TextProcessor`
used in our ``CustomAttackEval``.


Override Measure Method
-----------------------------

.. code-block:: python
    :linenos:

    def measure(self, x_orig, x_adv):
        info = super().measure(x_orig, x_adv)

        if info["Succeed"]:
            token_orig = [token for token, pos in self.__processor.get_tokens(x_orig)]
            token_adv = [token for token, pos in self.__processor.get_tokens(x_adv)]
            info["Bleu"] = sentence_bleu([x_orig], x_adv)

        return info

In this step, ``measure`` method is overriden.
It invokes the original ``measure`` method to get measurements and add ``Blue`` score which is calculated by **NLTK toolkit** if attack succeed.


Accumulate Measurements
------------------------------

.. code-block:: python
    :linenos:

    def update(self, info):
        info = super().update(info)
        if info["Succeed"]:
            self.__result["bleu"] += info["Bleu"]
        return info

``update`` method is used to accumulate results. 
We add bleu score that we just calculated to the total result.
Don't forget to call ``super().update(info)``.

Generate Summary
-------------------------

.. code-block:: python
    :linenos:

    def get_result(self):
        result = super().get_result()
        result["Avg. Bleu"] = self.__result["bleu"] / result["Successful Instances"]
        return result

The ``get_result`` method is called to generate a summary after all data is evaluated.
In this method, we calculate average bleu scores and return.

You can see :py:class:`.DefaultAttackEval` for more information.

Complete Code
--------------------------

.. code-block:: python
    :linenos:
    :name: examples/custom_measurement.py

    import OpenAttack
    from nltk.translate.bleu_score import sentence_bleu

    class CustomAttackEval(OpenAttack.DefaultAttackEval):
        def __init__(self, attacker, clsf, processor=OpenAttack.DefaultTextProcessor(), **kwargs):
            super().__init__(attacker, clsf, processor=processor, **kwargs)
            self.__processor = processor
        
        
        def measure(self, x_orig, x_adv):
            info = super().measure(x_orig, x_adv)

            if info["Succeed"]:
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

        attacker = OpenAttack.attackers.GeneticAttacker()
        attack_eval = CustomAttackEval(attacker, clsf)
        attack_eval.eval(dataset, visualize=True)

    if __name__ == "__main__":
        main()

Run ``python examples/custom_eval.py`` to see visualized results.