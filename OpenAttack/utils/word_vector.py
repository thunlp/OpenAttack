class WordVector:
    """
    This class is used to store word2id and word vector matrix.
    """
    def __init__(self, word2id, vec_matrix):
        self.id2vec = vec_matrix
        self.word2id = word2id

    def get_vecmatrix(self):
        """
        :return: Returns word vector matrix of shape (vocab_size, vector_dim).
        :rtype: nd.array
        """
        return self.id2vec

    def get_wordid(self, word):
        """
        :param str word: The word that you want to get index.
        :return: Word index, or None if not in vocabulary.
        :rtype: int or None
        """
        if word in self.word2id:
            return self.word2id[word]
        else:
            return None

    def get_dictionary(self):
        """
        :return: Returns a list of all words in vocabulary.
        :rtype: list
        """
        return list(self.word2id.keys())
