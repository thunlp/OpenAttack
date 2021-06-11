class Lemmatizer:
    def lemmatize(self, token : str, pos : str) -> str:
        return self.do_lemmatize(token, pos)
    
    def delemmatize(self, lemma : str, pos : str) -> str:
        return self.do_delemmatize(lemma, pos)
    
    def do_lemmatize(self, token, pos):
        raise NotImplementedError()
    
    def do_delemmatize(self, lemma, pos):
        raise NotImplementedError()
    