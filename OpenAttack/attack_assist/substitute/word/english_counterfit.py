from .embed_based import EmbedBasedSubstitute
from ....data_manager import DataManager
import torch

class CounterFittedSubstitute(EmbedBasedSubstitute):
    """
    :param bool cosine: If true, use cosine distance. **Default:** False.
    :Data Requirements: :py:data:`.AttackAssist.CounterFit`

    An implementation of :py:class:`.WordSubstitute`.

    Counter-fitting Word Vectors to Linguistic Constraints.
    `[pdf] <https://www.aclweb.org/anthology/N16-1018.pdf>`__
    """
    def __init__(self, cosine : bool = False, k : int = 50, threshold : float = 0.5, device = None):
        wordvec = DataManager.load("AttackAssist.CounterFit")

        super().__init__(
            wordvec.word2id,
            torch.from_numpy(wordvec.embedding),
            cosine = cosine,
            k = k,
            threshold = threshold,
            device = device
        )
