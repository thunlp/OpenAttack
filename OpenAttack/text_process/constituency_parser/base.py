class ConstituencyParser:
    def __call__(self, sentence : str) -> str:
        return self.parse(sentence)
    
    def parse(self, sentence : str) -> str:
        raise NotImplementedError()