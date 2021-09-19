============================================
Attacks on Chinese Dataset
============================================

In this example, we show how to use OpenAttack to attack your classifier on a Chinese dataset.

Load a Chinese Dataset
----------------------------
In this example, we read in the Chinese part of the `amazon_reviews_multi` dataset using datasets package.

.. code-block:: python
    :linenos:

    def dataset_mapping(x):
        return {
            "x": x["review_body"],
            "y": x["stars"],
        }
    dataset = datasets.load_dataset("amazon_reviews_multi",'zh',split="train[:5]").map(dataset_mapping)

We selected the first five examples in the dataset and used the `dataset_mapping` function to map the `review_body` field to `x` and the `stars` field to `y` which is the field victim models need to predict.

Load a Chinese Victim Model
----------------------------
OpenAttack provides a trained Chinese BERT classification model on the `amazon_reviews_multi` dataset.

In this example, we use `OpenAttack.loadVictim` to load the victim model and move it to a GPU.

.. code-block:: python
    :linenos:

    victim = OpenAttack.loadVictim("BERT.AMAZON_ZH").to("cuda:0")


Initialize an Attacker
-------------------------
Almost all `Attacker` take a `lang` argument which means the language you use.
We need to specify the `lang` argument before we start the attack.

.. code-block:: python
    :linenos:

    attacker = OpenAttack.attackers.PWWSAttacker(lang="chinese")

In this example, we chose the `PWWSAttacker` and set `lang` to `"chinese"`.

Use a Chinese AttackEval
-------------------------

The `OpenAttack` toolkit supports the `lang` option in `AttackEval` class to perform attack evaluation in different languages.

Just like the other examples, we only need one simple line of code to start the evaluation.

.. code-block:: python
    :linenos:

    OpenAttack.AttackEval(attacker, victim, lang="chinese").eval(dataset, progress_bar=True)


Complete Code 
-------------------

.. code-block:: python
    :linenos:

    import OpenAttack
    import datasets
    def main():
        attacker = OpenAttack.attackers.PWWSAttacker(lang="chinese")
        victim = OpenAttack.loadVictim("BERT.AMAZON_ZH").to("cuda:0")
        def dataset_mapping(x):
            return {
                "x": x["review_body"],
                "y": x["stars"],
            }
        dataset = datasets.load_dataset("amazon_reviews_multi",'zh',split="train[:20]").map(function=dataset_mapping)
        attack_eval = OpenAttack.AttackEval(attacker, victim)
        attack_eval.eval(dataset, visualize=True, progress_bar=True)

