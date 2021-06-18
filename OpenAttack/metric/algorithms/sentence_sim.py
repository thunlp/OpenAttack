from .base import AttackMetric
from ...tags import *

class SentenceSim(AttackMetric):
    
    NAME = "Sentence Similarity"
    TAGS = { TAG_English } 

    def __init__(self):
        """
        :Pakcage Requirements:
            * sentence_transformers
        :Language: english

        """
        from sentence_transformers import SentenceTransformer
        from ..data_manager import DataManager
        self.model = SentenceTransformer(DataManager.load("AttackAssist.SentenceTransformer"), device='cuda')

    def calc_score(self, sen1 : str, sen2 : str) -> float:
        """
        Args:
            sen1: The first sentence.
            sen2: The second sentence.
        Returns:
            Sentence similarity.
            
        """

        from sentence_transformers import util
        emb1,emb2 = self.model.encode([sen1,sen2],show_progress_bar=False)
        cos_sim = util.pytorch_cos_sim(emb1, emb2)
        return cos_sim.cpu().numpy()[0][0]
    
    def after_attack(self, input, adversarial_sample):
        if adversarial_sample is not None:
            return self.calc_score(input["x"], adversarial_sample)
