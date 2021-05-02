========================
Basic Usage
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

:py:data:`.Victim.BiLSTM.SST` is a pytorch model which is trained on SST dataset.
It uses :py:data:`Glove` vectors for word representation.

The load operation returns a :py:class:`.PytorchClassifier` that can be further used for *Attacker* and *AttackEval*.


Loading dataset
---------------------

.. code-block:: python

    def dataset_mapping(x):
        return {
            "x": x["sentence"],
            "y": 1 if x["label"] > 0.5 else 0,
        }
    dataset = datasets.load_dataset("sst").map(function=dataset_mapping)

We use the datasets package to manage our data, through which we load the data and map the fields to their corresponding places.

For each data instance, x means the attacked text and y means the true label of the data.


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
    import datasets
    def main():
        clsf = OpenAttack.DataManager.load("Victim.BiLSTM.SST")
        def dataset_mapping(x):
            return {
                "x": x["sentence"],
                "y": 1 if x["label"] > 0.5 else 0,
            }
        dataset = datasets.load_dataset("sst").map(function=dataset_mapping)

        attacker = OpenAttack.attackers.GeneticAttacker()
        attack_eval = OpenAttack.attack_evals.DefaultAttackEval(attacker, clsf)
        attack_eval.eval(dataset, visualize=True)

Run ``python examples/workflow.py`` to see visualized results.
