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
        """
        :param tf.keras.Model model: Tensorflow2 model for classification.
        :param str device: Device of tensorflow model. **Default:** "/cpu:0" if gpu is not available else "/gpu:0"
        :param TextProcessor processor: Text processor used for tokenization. **Default:** :any:`DefaultTextProcessor`
        :param dict word2id: A dict maps token to index. If it's not None, int64 EagerTensor will be passed to model. **Default:** None
        :param np.ndarray embedding: Word vector matrix of shape (vocab_size, vector_dim). If it's not None, float64 EagerTensor of shape (batch_size, max_input_len, vector_dim) will be passed to model.``word2id`` and ``embedding`` options are both required to support get_grad. **Default:** None
        :param int max_len: Max length of input tokens. If input token list is too long, it will be truncated. Uses None for no truncation. **Default:** None
        :param bool tokenization: If it's False, raw sentences will be passed to model, otherwise tokenized sentences will be passed. This option will be ignored if ``word2id`` is setted. **Default:** False
        :param bool padding: If it's True, add paddings to the end of sentences. This will be ignored if ``word2id`` option setted. **Default:** False
        :param str token_unk: Token for unknown tokens. **Default:** ``"<UNK>"``
        :param str token_unk: Token for padding. **Default:** ``"<PAD>"``
        :param bool require_length: If it's True, a list of lengths for each sentence will be passed to the model as the second parameter. **Default:** False

        :Package Requirements: * **tensorflow** >= 2.0.0
        """
        import tensorflow as tf

        self.model = model
        
        self.config = DEFAULT_CONFIG.copy()
        self.config["device"] = tf.device( "/gpu:0" if tf.test.is_gpu_available() else "/cpu:0" )
        self.config.update(kwargs)
        self.to(self.config["device"])

        check_parameters(DEFAULT_CONFIG.keys(), self.config)

        super().__init__(**self.config)
    
    def to(self, device):
        """
        :param str device: Device that moves model to.
        """
        if isinstance(device, str):
            import tensorflow as tf
            self.config["device"] = tf.device(device)
        else:
            self.config["device"] = device
        return self

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
        input_, seq_len = self.preprocess_token(input_)
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
