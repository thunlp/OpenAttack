import numpy as np

class UniversalSentenceEncoder:
    def __init__(self):
        """
        :Data Requirements: :py:data:`.AttackAssist.UniversalSentenceEncoder`
        :Package Requirements:
            * **tensorflow** >= 2.0.0
            * **tensorflow_hub**
        
        Universal Sentence Encoder in tensorflow_hub.
        `[pdf] <https://arxiv.org/pdf/1803.11175>`__
        `[page] <https://tfhub.dev/google/universal-sentence-encoder/4>`__
        """
        import logging
        import tensorflow as tf
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        tf.get_logger().setLevel(logging.ERROR)
        import tensorflow_hub as hub
        from ..data_manager import DataManager
        self.embed = hub.load( DataManager.load("AttackAssist.UniversalSentenceEncoder") )

    def __call__(self, sentA, sentB):
        """
        :param str sentA: The first sentence.
        :param str sentB: The second sentence.
        :return: Cosine distance between two sentences.
        :rtype: float
        """
        ret = self.embed([sentA, sentB]).numpy()
        return ret[0].dot(ret[1]) / (np.linalg.norm(ret[0]) * np.linalg.norm(ret[1]))
