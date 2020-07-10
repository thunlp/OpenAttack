import math
class GPT2LM:
    def __init__(self, use_tf=False):
        import transformers
        self.use_tf = use_tf

        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
        if use_tf:
            self.lm = transformers.TFGPT2LMHeadModel.from_pretrained("gpt2")
        else:
            self.lm = transformers.GPT2LMHeadModel.from_pretrained("gpt2")
    
    def __call__(self, sent):
        if self.use_tf:
            import tensorflow as tf
            ipt = self.tokenizer(sent, return_tensors="tf")
            ret = self.lm(ipt)[0]
            loss = 0
            for i in range(ret.shape[0]):
                it = ret[i]
                it = it - tf.reduce_max(it, axis=1)[:, tf.newaxis] 
                it = it - tf.math.log( tf.reduce_sum( tf.exp(it) , axis=1 ) )[:, tf.newaxis]
                it = tf.gather_nd(it, list(zip(range(it.shape[0] - 1), ipt.input_ids[i].numpy().tolist()[1:]) ) )
                loss += tf.reduce_mean( it )
                break
            return math.exp( -loss )
        else:
            ipt = self.tokenizer(sent, return_tensors="pt")
            return math.exp( self.lm(**ipt, labels=ipt.input_ids)[0] )
        

