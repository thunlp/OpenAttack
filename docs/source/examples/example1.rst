========================
OpenAttack Workflow
========================

This example shows a basic workflow of OpenAttack which consists of four main steps:
* Initializing classifier
* Loading dataset
* Initializing attacker
* Evaluation

Initializing Classifier
--------------------------

.. code-block:: python
    
    clsf = OpenAttack.DataManager.load("Victim.BiLSTM.SST")

:py:data:`.Victim.BiLSTM.SST` is a pytorch model which is trained on :py:data:`.Dataset.SST`.
It uses :py:data:`Glove` vectors for word representation.

The load operation returns a :py:class:`.PytorchClassifier` that can be further used for *Attacker* and *AttackEval*.


Loading dataset
---------------------

.. code-block:: python

    dataset = OpenAttack.DataManager.load("Dataset.SST.sample")

:py:data:`.Dataset.SST.sample` is a list of 1k sentences sampled from test dataset of :py:data:`.Dataset.SST`.


Initializing Attacker
----------------------

.. code-block:: python

    attacker = OpenAttack.attackers.GeneticAttacker()

After this step, we've initialized a :py:class:`.GeneticAttacker` and uses the default configuration during attack process.

To use a custom configuration, you can pass in some parameters manually.

Evaluation
-----------------------------

.. code-block:: python
    :linenos:

    attack_eval = OpenAttack.attack_evals.DefaultAttackEval(attacker, clsf)
    attack_eval.eval(dataset, visualize=True)

:py:class:`.DefaultAttackEval` is the default implementation for AttackEval which supports seven basic metrics.

Using ``visualize=True`` in `attack_eval.eval` can make it displays a visualized result.
This function is really useful for analyzing small datasets.

Complete Code
---------------------------

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
