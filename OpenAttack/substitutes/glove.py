from . import EmbedBasedSubstitute
from ..data_manager import DataManager


class GloveSubstitute(EmbedBasedSubstitute):
    """
    :param bool cosine: If true, use cosine distance. **Default:** False.
    :Data Requirements: :py:data:`.Glove`

    An implementation of :py:class:`.WordSubstitute`.
    """
    def __init__(self, cosine=False):
        self.wordvec = DataManager.load("AttackAssist.GloVe")
        self.word2id = {}
        for word in self.wordvec.get_dictionary():
            self.word2id[word] = len(self.word2id)

        super().__init__(
            cosine=cosine, embedding=self.wordvec.get_vecmatrix(), word2id=self.word2id
        )
