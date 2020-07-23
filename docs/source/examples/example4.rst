============================================
Use Additional Constraints
============================================

In this example, we write a new AttackEval to evaluate attackers with the constraint of grammatical errors.
This example is a bit harder than the other tree examples because a deeper understanding of OpenAttack is needed.

Initialize AttackEval
-----------------------

.. code-block:: python
    :linenos:

    class AttackEvalConstraint(OpenAttack.attack_evals.DefaultAttackEval):
        def __init__(self, attacker, clsf, mistake_limit=5, **kwargs):
            self.mistake_limit = mistake_limit
            super().__init__(attacker, clsf, mistake=True, **kwargs)

We extend :py:class:`.DefaultAttackEval` and set mistake option to be True.
This makes :py:class:`.DefaultAttackEval` to measure grammatical errors. (**package requirements:** language_tool_python)


Override Measure Method
-----------------------------

.. code-block:: python
    :linenos:

    def measure(self, sentA, sentB):
        info = super().measure(sentA, sentB)
        if info["Succeed"] and info["Grammatical Errors"] >= self.mistake_limit:
            info["Succeed"] = False
        return info

In this step, ``measure`` method is overriden.
It invokes the original ``measure`` method to get measurements and checks if grammatical erros is grater than ``mistake_limit``.

You can see :py:class:`.DefaultAttackEval` for more information.

Complete Code
--------------------------

.. code-block:: python
    :linenos:
    :name: examples/custom_constraint.py

    import OpenAttack
    class AttackEvalConstraint(OpenAttack.attack_evals.DefaultAttackEval):
        def __init__(self, attacker, clsf, mistake_limit=5, **kwargs):
            self.mistake_limit = mistake_limit
            super().__init__(attacker, clsf, mistake=True, **kwargs)
        
        def measure(self, sentA, sentB):
            info = super().measure(sentA, sentB)
            if info["Succeed"] and info["Grammatical Errors"] >= self.mistake_limit:
                info["Succeed"] = False
            return info
    def main():
        clsf = OpenAttack.load("Victim.BiLSTM.SST")
        dataset = OpenAttack.load("Dataset.SST.sample")[:10]
        attacker = OpenAttack.attackers.PWWSAttacker()
        attack_eval = AttackEvalConstraint(attacker, clsf)
        attack_eval.eval(dataset, visualize=True)

Run ``python examples/custom_constraint.py`` to see visualized results.