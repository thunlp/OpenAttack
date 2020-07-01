import numpy as np

class UniversalSentenceEncoder:
    def __init__(self):
        import tensorflow_hub as hub
        from ..data_manager import DataManager
        self.embed = hub.load( DataManager.load("UniversalSentenceEncoder") )

    def __call__(self, sentA, sentB):
        ret = self.embed([sentA, sentB]).numpy()
        return ret[0].dot(ret[1]) / (np.linalg.norm(ret[0]) * np.linalg.norm(ret[1]))