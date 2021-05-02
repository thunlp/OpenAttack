import math


class GPT2LMCH:
    def __init__(self):
        """
        :Package Requirements:
            * **torch** (if use_tf = False)
            * **tensorflow** >= 2.0.0 (if use_tf = True)
            * **transformers**
        """
        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import transformers

        self.tokenizer = transformers.BertTokenizer.from_pretrained("mymusise/EasternFantasyNoval")
        self.lm = transformers.TFGPT2LMHeadModel.from_pretrained("mymusise/EasternFantasyNoval")

    def __call__(self, sent):
        """
        :param str sent: A sentence.
        :return: Fluency (ppl).
        :rtype: float
        """
        import tensorflow as tf
        ipt = self.tokenizer(sent, return_tensors="tf", verbose=False)
        ret = self.lm(ipt)[0]
        loss = 0
        for i in range(ret.shape[0]):
            it = ret[i]
            it = it - tf.reduce_max(it, axis=1)[:, tf.newaxis]
            it = it - tf.math.log(tf.reduce_sum(tf.exp(it), axis=1))[:, tf.newaxis]
            it = tf.gather_nd(it, list(zip(range(it.shape[0] - 1), ipt.input_ids[i].numpy().tolist()[1:])))
            loss += tf.reduce_mean(it)
            break
        return math.exp(-loss)
