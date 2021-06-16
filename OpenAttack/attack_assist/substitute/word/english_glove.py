import torch
from .embed_based import EmbedBasedSubstitute
from ....data_manager import DataManager
from ....tags import TAG_English


class GloveSubstitute(EmbedBasedSubstitute):
    """
    :param bool cosine: If true, use cosine distance. **Default:** False.
    :Data Requirements: :py:data:`.Glove`

    An implementation of :py:class:`.WordSubstitute`.
    """

    TAGS = { TAG_English } 

    def __init__(self, cosine = False, k = 50, threshold = 0.5, device = None):
        wordvec = DataManager.load("AttackAssist.GloVe")
        
        super().__init__(
            wordvec.word2id,
            torch.from_numpy(wordvec.embedding),
            cosine = cosine, 
            k = k,
            threshold = threshold,
            device = device
        )
