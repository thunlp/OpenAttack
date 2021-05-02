============================================
Attacks on Chinese Dataset
============================================

In this example, we show how to use OpenAttack to attack your classifier on a Chinese dataset with a customized `TextProcessor`.


Build a Chinese TextProcessor
-------------------------------
Normally, if you need to use your own text processing methods you need to derive the TextProcessor class.
The most important part of the TextProcessor class is the word tokenization and POS tagging.
For most attackers, you need to implement the get_tokens interface, which takes a sentence as input and returns a list of (token, pos).

.. code-block:: python
    :linenos:

    class ChineseTextProcessor(TextProcessor):
        def get_tokens(self, sentence):
            import jieba
            import jieba.posseg as pseg
            mapping = {
                'v': 'VBD',
                'n': 'NN',
                'r': 'PRP',
                't': 'NN',
                'm': 'DT',
                'f': 'IN',
                'a': 'JJ',
                'd': 'RB'
            }
            ans = []
            for pair in pseg.cut(sentence):  
                if pair.flag[0] in mapping:
                    ans.append((pair.word, mapping[ pair.flag[0] ]))
                else:
                    ans.append((pair.word, "OTHER" ))
            return ans

In this example, we use the jieba toolkit to perform tokenization and POS tagging.
Note that the result POS tagset need to be mapped to Penn tagset.


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

    clsf = OpenAttack.loadVictim("BERT.AMAZON_ZH").to("cuda:0")


Initialize an Attacker
-------------------------
Almost all `Attacker` take the `processor` as an argument.
In addition to the `processor`, each `attacker` may have some other arguments, such as the `substitute`, etc. 
We need to specify these arguments before we start the attack.

.. code-block:: python
    :linenos:

    chinese_processor = ChineseTextProcessor()
    chinese_substitute = OpenAttack.substitutes.ChineseHowNetSubstitute()
    attacker = OpenAttack.attackers.PWWSAttacker(processor=chinese_processor, substitute=chinese_substitute, threshold=10)

In this example, we have chosen the `PWWSAttacker` method and set the values of `processor` and `substitute`.

Use a Chinese AttackEval
-------------------------

The `OpenAttack` toolkit provides a Chinese `AttackEval` class to perform Chinese attack evaluation.

Just like the other examples, we only need one simple line of code to start the evaluation.

.. code-block:: python
    :linenos:

    OpenAttack.attack_evals.ChineseAttackEval(attacker, clsf).eval(dataset, progress_bar=True)

