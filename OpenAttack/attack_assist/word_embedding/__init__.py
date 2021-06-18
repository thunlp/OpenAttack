
from typing import Dict


class WordEmbedding:
    def __init__(self, word2id : Dict[str, int], embedding) -> None:
        self.word2id = word2id
        self.embedding = embedding
    
    def transform(self, word, token_unk):
        if word in self.word2id:
            return self.embedding[ self.word2id[word] ]
        else:
            if isinstance(token_unk, int):
                return self.embedding[ token_unk ]
            else:
                return self.embedding[ self.word2id[ token_unk ] ]