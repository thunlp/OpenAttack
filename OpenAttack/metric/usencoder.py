import numpy as np

class UniversalSentenceEncoder:
    def __init__(self):
        import logging
        import tensorflow as tf
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        tf.get_logger().setLevel(logging.ERROR)
        import tensorflow_hub as hub
        from ..data_manager import DataManager
        self.embed = hub.load( DataManager.load("UniversalSentenceEncoder") )

    def __call__(self, sentA, sentB):
        ret = self.embed([sentA, sentB]).numpy()
        return ret[0].dot(ret[1]) / (np.linalg.norm(ret[0]) * np.linalg.norm(ret[1]))
