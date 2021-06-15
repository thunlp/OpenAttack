from .base import AttackMetric
import numpy as np
from ...tags import *
from ...data_manager import DataManager

## TODO use a pytorch model instead

class UniversalSentenceEncoder(AttackMetric):

    NAME = "Semantic Similarity"

    TAGS = { TAG_English }

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
        
        import tensorflow_hub as hub
        
        self.embed = hub.load( DataManager.load("AttackAssist.UniversalSentenceEncoder") )

    def calc_score(self, sentA, sentB):
        """
        :param str sentA: The first sentence.
        :param str sentB: The second sentence.
        :return: Cosine distance between two sentences.
        :rtype: float
        """
        ret = self.embed([sentA, sentB]).numpy()
        return ret[0].dot(ret[1]) / (np.linalg.norm(ret[0]) * np.linalg.norm(ret[1]))

    def after_attack(self, input, adversarial_sample):
        if adversarial_sample is not None:
            return self.calc_score(input["x"], adversarial_sample)
