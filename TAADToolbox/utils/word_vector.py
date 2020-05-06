class WordVector:
    def __init__(self, word2id, vec_matrix):
        self.id2vec = vec_matrix
        self.word2id = word2id

    def get_vecmatrix(self):
        return self.id2vec

    def get_wordid(self, word):
        if word in self.word2id:
            return self.word2id[word]
        else:
            return None

    def get_dictionary(self):
        return list(self.word2id.keys())
