============================
Using A Custom Classifier
============================

In this example, we will show you how to adapt to your own classifer.

``nltk.sentiment.vader.SentimentIntensityAnalyzer`` is a tradictional sentiment classification model.
We writes a special :py:class:`.Classifier` for it and applies PWWSAttacker on our new classifier.
See the code below.

Write a New Classifier
-------------------------

.. code-block:: python
    :linenos:

    import numpy as np
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    class MyClassifier(OpenAttack.Classifier):
        def __init__(self):
            self.model = SentimentIntensityAnalyzer()

        def get_prob(self, input_):
            ret = []
            for sent in input_:
                res = self.model.polarity_scores(sent)
                prob = (res["pos"] + 1e-6) / (res["neg"] + res["pos"] + 2e-6)
                ret.append(np.array([1 - prob, prob]))
            return np.array(ret)

Firstly, a new classifier is implemented by extending :py:class:`OpenAttack.Classifier`.
``SentimentIntensityAnalyzer`` calculates scores of "neg" and "pos" for each instance,
and we use :math:`\frac{socre_{pos}}{score_{neg} + score_{pos}}` to represent the probability of positive sentiment.
Adding :math:`10^{-6}` is a trick to avoid dividing by zero.

The ``get_prob`` method finally returns a ``np.ndarray`` of shape (len(input\_), 2). See :py:class:`.Classifier` for detail.

Load Dataset and Evaluate
------------------------------
.. code-block:: python
    :linenos:
    
    dataset = OpenAttack.load("Dataset.SST.sample")[:10]
    clsf = MyClassifier()
    attacker = OpenAttack.attackers.PWWSAttacker()
    attack_eval = OpenAttack.attack_evals.DefaultAttackEval(attacker, clsf)
    attack_eval.eval(dataset, visualize=True)

Secondly, we load :py:data:`.Dataset.SST.sample` for evaluation and initialize ``MyClassifier`` which is defined in the first step.
Then :py:class:`.PWWSAttacker` is initialized, this attacker requires classifier to be able to generate probability for each label.
It's worthy to note that all classifiers are divid into three different level of capacity -- "Blind", "Probability" and "Gradient".
*"Blind"* means classifier can only predict labels, *"Probability"* means classifier predict probability for each label and "Gradient" means besides probability, gradient is also accessible.

We implemented ``get_prob`` in the first step and :py:class:`.PWWSAttacker` needs a capacity of "Probability",
so our classifier can be correctly applied to :py:class:`.PWWSAttacker`.

Complete Code
----------------

.. code-block:: python
    :linenos:
    :name: examples/custom_classifier.py
    
    import OpenAttack
    import numpy as np
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    class MyClassifier(OpenAttack.Classifier):
        def __init__(self):
            self.model = SentimentIntensityAnalyzer()

        def get_prob(self, input_):
            ret = []
            for sent in input_:
                res = self.model.polarity_scores(sent)
                prob = (res["pos"] + 1e-6) / (res["neg"] + res["pos"] + 2e-6)
                ret.append(np.array([1 - prob, prob]))
            return np.array(ret)
            
    def main():
        dataset = OpenAttack.load("Dataset.SST.sample")[:10]

        clsf = MyClassifier()
        attacker = OpenAttack.attackers.PWWSAttacker()
        attack_eval = OpenAttack.attack_evals.DefaultAttackEval(attacker, clsf)
        attack_eval.eval(dataset, visualize=True)

Run ``python examples/custom_classifier.py`` to see visualized results.