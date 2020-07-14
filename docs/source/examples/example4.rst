============================================
Evaluate Attacker with Custom Constraints
============================================

xxxxx

.. code-block:: python
    :linenos:

    class AttackEvalConstraint(OpenAttack.attack_evals.DefaultAttackEval):
        def __init__(self, attacker, clsf, mistake_limit=5, **kwargs):
            self.mistake_limit = mistake_limit
            super().__init__(attacker, clsf, mistake=True, **kwargs)

xxxx

.. code-block:: python
    :linenos:
    :name: examples/custom_constraint.py

    def measure(self, sentA, sentB):
        info = super().measure(sentA, sentB)
        if info["Succeed"] and info["Grammatical Errors"] >= self.mistake_limit:
            info["Succeed"] = False
        return info

xxxx

.. code-block:: python
    :linenos:
    
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

xxx