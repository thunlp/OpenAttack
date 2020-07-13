import numpy as np
from . import ClassifierBase
from ..text_processors import DefaultTextProcessor
from ..utils import check_parameters
from ..exceptions import ClassifierNotSupportException

DEFAULT_CONFIG = {
    "device": None,
    "processor": DefaultTextProcessor(),
    "embedding": None,
    "word2id": None,
    "max_len": None,
    "tokenization": False,
    "padding": False,
    "token_unk": "<UNK>",
    "token_pad": "<PAD>",
    "require_length": False
}



class TensorflowClassifier(ClassifierBase):
    def __init__(self, model, **kwargs):
        import tensorflow as tf

        self.model = model
        
        self.config = DEFAULT_CONFIG.copy()
        self.config["device"] = tf.device( "/gpu:0" if tf.test.is_gpu_available() else "/cpu:0" )
        self.config.update(kwargs)
        check_parameters(DEFAULT_CONFIG.keys(), self.config)

        super().__init__(**self.config)

    def get_pred(self, input_):
        import tensorflow as tf

        input_, seq_len = self.preprocess(input_)
        if isinstance(input_, np.ndarray):
            with self.config["device"]:
                input_ = tf.constant( input_ )
        if self.config["tokenization"] and self.config["require_length"]:
            prob = self.model(input_, seq_len)
        else:
            prob = self.model(input_)
        return tf.math.argmax(prob, 1).numpy()

    def get_prob(self, input_):
        import tensorflow as tf
        input_, seq_len = self.preprocess(input_)
        if isinstance(input_, np.ndarray):
            with self.config["device"]:
                input_ = tf.constant( input_ )

        if self.config["tokenization"] and self.config["require_length"]:
            prob = self.model(input_, seq_len)
        else:
            prob = self.model(input_)
        return prob.numpy()

    def get_grad(self, input_, labels):
        if self.config["word2id"] is None or self.config["embedding"] is None:
            raise ClassifierNotSupportException("gradient")

        import tensorflow as tf
        input_, seq_len = self.preprocess(input_)
        if isinstance(input_, np.ndarray):
            with self.config["device"]:
                input_ = tf.constant( input_ )
        
        with tf.GradientTape() as t:
            t.watch(input_)
            if self.config["require_length"]:
                prob = self.model(input_, seq_len)
            else:
                prob = self.model(input_)
            loss = tf.reduce_sum(tf.gather_nd(prob, list( zip( range(len(labels)), labels ))))
        gradient = t.gradient(loss, input_)
        return prob.numpy(), gradient.numpy()
