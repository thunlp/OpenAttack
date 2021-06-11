import unittest
import os
from OpenAttack import DataManager


class TestDataManager(unittest.TestCase):
    def test_tokenize(self):
        from OpenAttack.text_process.tokenizer import PunctTokenizer
        tokenizer = PunctTokenizer()
        ret = tokenizer.tokenize("This is an apple.")
        self.assertIsInstance(ret, list)
        self.assertGreater(len(ret), 0)
        for i in range(len(ret)):
            self.assertEqual(len(ret[i]), 2)
            self.assertIsInstance(ret[i][0], str)
            self.assertIsInstance(ret[i][1], str)
        self.assertIsInstance(tokenizer.detokenize(ret), str)

    def test_lemma(self):
        from OpenAttack.text_process.lemmatizer import WordnetLemmatimer
        
        lemmatizer = WordnetLemmatimer()
        test_cases = [('There', 'other'), ('were', 'verb'), ('apples', 'noun'), ('.', 'other')]
        for case in test_cases:
            ret = lemmatizer.lemmatize(case[0], case[1])

            self.assertIsInstance(ret, str)

        ret = lemmatizer.delemmatize("apples", "noun")
        self.assertEqual(ret, "apples")
    
        ret = lemmatizer.delemmatize("apple", "noun")
        self.assertEqual(ret, "apples")
    
    
    """
    def test_parser(self):
        from OpenAttack.text_process.constituency_parser import StanfordParser
        parser = StanfordParser()
        ret = parser("The quick brown fox jumps over a lazy dog.")
        self.assertIsInstance(ret, str)
    """