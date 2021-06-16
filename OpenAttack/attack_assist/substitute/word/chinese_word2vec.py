from .embed_based import EmbedBasedSubstitute
from ....data_manager import DataManager
from ....tags import TAG_Chinese
import torch

class ChineseWord2VecSubstitute(EmbedBasedSubstitute):
    """
    :param bool cosine: If true, use cosine distance. **Default:** False.
    :Data Requirements: :py:data:`.AttackAssist.ChineseWord2Vec`

    An implementation of :py:class:`.WordSubstitute`.
    """

    TAGS = { TAG_Chinese }

    def __init__(self, cosine=False, threshold = 0.5, k = 50, device = None):
        wordvec = DataManager.load("AttackAssist.ChineseWord2Vec")

        super().__init__(
            wordvec.word2id,
            embedding = torch.from_numpy(wordvec.embedding), 
            cosine = cosine, 
            k = k,
            threshold = threshold,
            device = device
        )
