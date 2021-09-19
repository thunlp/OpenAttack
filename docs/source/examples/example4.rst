============================================
Customized Measurement
============================================

In this example, we write a new AttackMetric to evaluate attackers and report average input length.

Write a New AttackMetric
---------------------------

.. code-block:: python
    :linenos:

    class SentenceLength(OpenAttack.AttackMetric):
        NAME = "Input Length"

        def after_attack(self, input, adversarial_sample):
            return len(input["x"].split(" "))


We extend :py:class:`.AttackMetric` and override `after_attack` method to report the length of attacked sentence (assume that words are separated by white spaces).

``NAME`` attribute  indicates the name of the ``AttackMetric`` that will show out in the final result.


Apply AttackMetrics in AttackEval
----------------------------------

.. code-block:: python
    :linenos:

    attack_eval = OpenAttack.AttackEval(attacker, clsf, metrics=[
        SentenceLength()
    ])

``AttackEval`` supports the ``metrics`` option to specify evaluation metrics.

There are some built-in metrics in ``OpenAttack``:

* OpenAttack.metric.Fluency()
* OpenAttack.metric.GrammaticalErrors()
* OpenAttack.metric.SemanticSimilarity()
* OpenAttack.metric.EditDistance()
* OpenAttack.metric.ModificationRate()

These metrics are common in most attack evaluations.

Complete Code
--------------------------

.. code-block:: python
    :linenos:
    :name: examples/custom_measurement.py

    import OpenAttack
    import datasets
    class SentenceLength(OpenAttack.AttackMetric):
        NAME = "Input Length"
        def after_attack(self, input, adversarial_sample):
            return len(input["x"].split(" "))
    def main():
        victim = OpenAttack.loadVictim("BERT.SST")
        def dataset_mapping(x):
            return {
                "x": x["sentence"],
                "y": 1 if x["label"] > 0.5 else 0,
            }
        dataset = datasets.load_dataset("sst", split="train[:20]").map(function=dataset_mapping)
        attacker = OpenAttack.attackers.GeneticAttacker()
        attack_eval = OpenAttack.AttackEval(attacker, victim, metrics=[
            SentenceLength()
        ])
        attack_eval.eval(dataset, visualize=True)



Run ``python examples/custom_metrics.py`` to see visualized results.