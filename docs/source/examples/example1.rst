========================
OpenAttack Workflow
========================

xxx

.. code-block:: python
    
    clsf = OpenAttack.DataManager.load("Victim.BiLSTM.SST")

xxx

.. code-block:: python

    dataset = OpenAttack.DataManager.load("Dataset.SST.sample")

xxx

.. code-block:: python

    attacker = OpenAttack.attackers.GeneticAttacker()

xxx

.. code-block:: python
    :linenos:

    attack_eval = OpenAttack.attack_evals.DefaultAttackEval(attacker, clsf)
    attack_eval.eval(dataset, visualize=True)

xxx

.. code-block:: python
    :linenos:
    :name: examples/workflow.py

    import OpenAttack
    def main():
        clsf = OpenAttack.DataManager.load("Victim.BiLSTM.SST")
        dataset = OpenAttack.DataManager.load("Dataset.SST.sample")[:10]

        attacker = OpenAttack.attackers.GeneticAttacker()
        attack_eval = OpenAttack.attack_evals.DefaultAttackEval(attacker, clsf)
        attack_eval.eval(dataset, visualize=True)

xxxx
