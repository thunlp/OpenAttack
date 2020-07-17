============================
Evaluate A New Attacker
============================

Attacker is the core module of OpenAttack. In this example, we write a new attacker which swaps tokens
with the same POS. This is a simple way to generate adversarial samples and it requires a capacity of "Blind".


Initialize Attacker with Options
----------------------------------

.. code-block:: python
    :linenos:
    
    class MyAttacker(OpenAttack.Attacker):
        def __init__(self, max_iter=20, processor = OpenAttack.text_processors.DefaultTextProcessor()):
            self.processor = processor
            self.max_iter = max_iter

We add two parameters for our new attacker, ``max_iter`` means the maximum number of iterations to swap tokens
and ``processor`` is used for tokenization. Providing default value for each parameter is a good behavior in OpenAttack.
This makes it easier for users to use, and also makes your new attacker easier to be integrated into OpenAttack.

Randomly Swap Tokens with Same POS
----------------------------------------

.. code-block:: python
    :linenos:

    def swap(self, sent_token):
        pairs = []
        for i in range(len(sent_token)):
            for j in range(i):
                if sent_token[i][1] == sent_token[j][1]:
                    pairs.append((i, j))
        if len(pairs) == 0:
            return sent_token

        import random
        pi, pj = random.choice(pairs)   # random select one pair
        sent_token[pi], sent_token[pj] = sent_token[pj], sent_token[pi] # swap this pair
        return sent_token

In ``swap`` method, we record all pairs with the same POS in ``pairs`` list, then randomly choose one from it and make a swap.

Major Procedures of Attack
---------------------------------

.. code-block:: python
    :linenos:

    def __call__(self, clsf, x_orig, target=None):
        if target is None:
            target = clsf.get_pred([x_orig])[0]
            targeted = False
        else:
            targeted = True
        
        # generate samples
        all_sents = []
        curr_x = self.processor.get_tokens(x_orig)
        for i in range(self.max_iter):
            curr_x = self.swap(curr_x)
            sent = OpenAttack.utils.detokenizer(curr_x)
            all_sents.append(sent)
        
        # get prediction
        preds = clsf.get_pred(all_sents)

        for idx, sent in enumerate(all_sents):
            if targeted:
                if preds[idx] == target:
                    return (sent, preds[idx])
            else:
                if preds[idx] != target:
                    return (sent, preds[idx])
        return None

``target`` parameter can be either None or int. Int is for target attack while None for untargeted attack.
We generate ``max_iter`` sentences, get predictions from classifier, 
and check each prediction to find a succeed one then return.  If attacker is failed, ``__call__`` returns None.

See :py:class:`.Attacker` for detail.

Complete Code
------------------------------

.. code-block:: python
    :linenos:
    :name: examples/custom_attacker.py
    
    import OpenAttack
    class MyAttacker(OpenAttack.Attacker):
        def __init__(self, max_iter=20, processor = OpenAttack.text_processors.DefaultTextProcessor()):
            self.processor = processor
            self.max_iter = max_iter 
        def __call__(self, clsf, x_orig, target=None):
            if target is None:
                target = clsf.get_pred([x_orig])[0]
                targeted = False
            else:
                targeted = True
            # generate samples
            all_sents = []
            curr_x = self.processor.get_tokens(x_orig)
            for i in range(self.max_iter):
                curr_x = self.swap(curr_x)
                sent = OpenAttack.utils.detokenizer(curr_x)
                all_sents.append(sent)
            # get prediction
            preds = clsf.get_pred(all_sents)
            for idx, sent in enumerate(all_sents):
                if targeted:
                    if preds[idx] == target:
                        return (sent, preds[idx])
                else:
                    if preds[idx] != target:
                        return (sent, preds[idx])
            return None
        def swap(self, sent_token):
            pairs = []
            for i in range(len(sent_token)):
                for j in range(i):
                    if sent_token[i][1] == sent_token[j][1]:    # same POS
                        pairs.append((i, j))
            if len(pairs) == 0:
                return sent_token
            import random
            pi, pj = random.choice(pairs)   # random select one pair
            sent_token[pi], sent_token[pj] = sent_token[pj], sent_token[pi] # swap this pair
            return sent_token

    def main():
        clsf = OpenAttack.DataManager.load("Victim.BiLSTM.SST")
        dataset = OpenAttack.DataManager.load("Dataset.SST.sample")[:10]
        attacker = MyAttacker()
        attack_eval = OpenAttack.attack_evals.DefaultAttackEval(attacker, clsf)
        attack_eval.eval(dataset, visualize=True)

Run ``python examples/custom_attacker.py`` to see visualized results.