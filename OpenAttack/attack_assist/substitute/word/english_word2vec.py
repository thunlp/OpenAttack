from .embed_based import EmbedBasedSubstitute
from ....data_manager import DataManager
import torch

class Word2VecSubstitute(EmbedBasedSubstitute):
    """
    :param bool cosine: If true, use cosine distance. **Default:** False.
    :Data Requirements: :py:data:`.Word2Vec`

    An implementation of :py:class:`.WordSubstitute`.
    """
    def __init__(self, cosine = False, k = 50, threshold = 0.5, device = None):
        wordvec = DataManager.load("AttackAssist.Word2Vec")

        super().__init__(
            wordvec.word2id,
            torch.from_numpy(wordvec.embedding),
            cosine = cosine,
            k = k,
            threshold = threshold,
            device = device
        )
