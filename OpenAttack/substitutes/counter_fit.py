from . import EmbedBasedSubstitute
from ..data_manager import DataManager


class CounterFittedSubstitute(EmbedBasedSubstitute):
    def __init__(self, cosine=False):
        self.wordvec = DataManager.load("CounterFit")
        self.word2id = {}
        for word in self.wordvec.get_dictionary():
            self.word2id[word] = len(self.word2id)

        super().__init__(
            cosine=cosine, embedding=self.wordvec.get_vecmatrix(), word2id=self.word2id
        )
