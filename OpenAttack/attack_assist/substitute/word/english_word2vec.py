from .embed_based import EmbedBasedSubstitute
from ....data_manager import DataManager
from ....tags import TAG_English
import torch

class Word2VecSubstitute(EmbedBasedSubstitute):

    TAGS = { TAG_English }

    def __init__(self, cosine = False, k = 50, threshold = 0.5, device = None):
        """
        English word substitute based on word2vec.

        Args:
            cosine: If `true` then the cosine distance is used, otherwise the Euclidian distance is used.
            threshold: Distance threshold. Default: 0.5
            k: Top-k results to return. If k is `None`, all results will be returned. Default: 50
            device: A pytocrh device for computing distances. Default: "cpu"
        
        :Data Requirements: :py:data:`.AttackAssist.GloVe`
        :Language: english
        
        """

        wordvec = DataManager.load("AttackAssist.Word2Vec")

        super().__init__(
            wordvec.word2id,
            torch.from_numpy(wordvec.embedding),
            cosine = cosine,
            k = k,
            threshold = threshold,
            device = device
        )
