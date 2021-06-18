from typing import Union
from .embed_based import EmbedBasedSubstitute
from ....data_manager import DataManager
from ....tags import TAG_Chinese
import torch

class ChineseWord2VecSubstitute(EmbedBasedSubstitute):

    TAGS = { TAG_Chinese }

    def __init__(self, cosine : bool = False, threshold : float = 0.5, k : int = 50, device : Union[str, torch.device, None] = None):
        """
        Chinese word substitute based on word2vec.

        Args:
            cosine: If `true` then the cosine distance is used, otherwise the Euclidian distance is used.
            threshold: Distance threshold. Default: 0.5
            k: Top-k results to return. If k is `None`, all results will be returned. Default: 50
            device: A pytocrh device for computing distances. Default: "cpu"
        
        :Data Requirements: :py:data:`.AttackAssist.ChineseWord2Vec`
        :Language: chinese
        
        """

        wordvec = DataManager.load("AttackAssist.ChineseWord2Vec")

        super().__init__(
            wordvec.word2id,
            embedding = torch.from_numpy(wordvec.embedding), 
            cosine = cosine, 
            k = k,
            threshold = threshold,
            device = device
        )
