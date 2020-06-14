import numpy as np
from .pre_processor import PreProcessor
from ..classifier import Classifier
from ..text_processors import DefaultTextProcessor
from ..utils import check_parameters
from ..exceptions import ClassifierNotSupportException

DEFAULT_CONFIG = {
    "device": 'CPU:0',
    "processor": DefaultTextProcessor(),
    "embedding": None,
    "vocab": None,
    "max_len":None,    
    "use_sentence": False,
    "use_word_id": False,
    "use_embedding": True,
}


class TensorflowClassifier(Classifier):
    def __init__(self, *args, **kwargs):
        import tensorflow as tf

        self.model = args[0]
        
        self.config = DEFAULT_CONFIG.copy()
        self.config.update(kwargs)
        check_parameters(DEFAULT_CONFIG.keys(), self.config)
        self.use_sentence = self.config["use_sentence"]
        self.use_word_id = self.config["use_word_id"]
        self.use_embedding = self.config["use_embedding"]
        if self.use_word_id or self.use_embedding:
            self.pre_processor = PreProcessor(self.config["vocab"], self.config["max_len"], processor=self.config["processor"], embedding=self.config["embedding"])

    def get_pred(self, input_):
        import tensorflow as tf

        if self.use_sentence:
            prob = self.model(input_)
        elif self.use_word_id:
            with tf.device(self.config["device"]):
                seqs = tf.constant(self.pre_processor.POS_process(input_))
            prob = self.model(seqs)
        elif self.use_embedding:
            seqs = self.pre_processor.POS_process(input_)
            
            with tf.device(self.config["device"]):
                seqs2 = tf.constant(self.pre_processor.embedding_process(seqs))
            prob = self.model(seqs2)
        return tf.math.argmax(prob, 1).numpy()
        

    def get_prob(self, input_):
        import tensorflow as tf

        if self.use_sentence:
            prob = self.model(input_)
        elif self.use_word_id:
            with tf.device(self.config["device"]):
                seqs = tf.constant(self.pre_processor.POS_process(input_))
            prob = self.model(seqs)
        elif self.use_embedding:
            seqs = self.pre_processor.POS_process(input_)
            with tf.device(self.config["device"]):
                seqs2 = tf.constant(self.pre_processor.embedding_process(seqs))
            prob = self.model(seqs2)
        return prob.numpy()

    def get_grad(self, input_, labels):
        import tensorflow as tf

        if self.use_sentence:
            raise ClassifierNotSupportException
        elif self.use_word_id:
            raise ClassifierNotSupportException
        elif self.use_embedding:
            seqs = self.pre_processor.POS_process(input_)
            with tf.device(self.config["device"]):
                seqs2 = tf.constant(self.pre_processor.embedding_process(seqs))
            with tf.GradientTape() as t:
                t.watch(seqs2)
                prob = self.model(seqs2)
                loss = tf.zeros([1])
                for i in range(len(labels)):
                    loss += prob[i][labels[i]]
            gradient = t.gradient(loss, seqs2)
        return gradient.numpy()