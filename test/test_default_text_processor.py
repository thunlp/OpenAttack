import unittest
import OpenAttack
import os


class TestDataManager(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OpenAttack.DataManager.set_path("./testdir")
        OpenAttack.DataManager.download("NLTKSentTokenizer")
        OpenAttack.DataManager.download("NLTKPerceptronPosTagger")
        OpenAttack.DataManager.download("NLTKWordnet")
        OpenAttack.DataManager.download("NLTKWordnetDelemma")
        OpenAttack.DataManager.download("StanfordNER")
        OpenAttack.DataManager.download("StanfordParser")
        cls.dp = OpenAttack.text_processors.DefaultTextProcessor()
    
    @classmethod
    def tearDownClass(cls):
        os.system("rm -r ./testdir")

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_tokenize(self):
        ret = self.dp.get_tokens("This is an apple.")
        self.assertIsInstance(ret, list)
        self.assertGreater(len(ret), 0)
        for i in range(len(ret)):
            self.assertEqual(len(ret[i]), 2)
            self.assertIsInstance(ret[i][0], str)
            self.assertIsInstance(ret[i][1], str)

    def test_lemma(self):
        ret = self.dp.get_lemmas([('There', 'EX'), ('were', 'VBD'), ('apples', 'NNS'), ('.', '.')])
        self.assertIsInstance(ret, list)
        self.assertGreater(len(ret), 0)
        for i in range(len(ret)):
            self.assertIsInstance(ret[i], str)
        ret = self.dp.get_lemmas(("apples", "NNS"))
        self.assertEqual(ret, "apple")
    
    def test_delemma(self):
        ret = self.dp.get_delemmas(("apple", "NNS"))
        self.assertEqual(ret, "apples")
    
    def test_ner(self):
        ret = self.dp.get_ner("New York is the biggest city in America.")
        self.assertIsInstance(ret, list)
        self.assertGreater(len(ret), 0)
        for i in range(len(ret)):
            self.assertEqual(len(ret[i]), 4)
            self.assertIsInstance(ret[i][0], str)
            self.assertIsInstance(ret[i][1], int)
            self.assertIsInstance(ret[i][2], int)
            self.assertIsInstance(ret[i][3], str)
    
    def test_parser(self):
        ret = self.dp.get_parser("The quick brown fox jumps over a lazy dog.")
        self.assertIsInstance(ret, str)
    
    def test_wsd(self):
        ret = self.dp.get_wsd([
            ('I', 'PRP'), 
            ('went', 'VBD'), 
            ('to', 'TO'), 
            ('the', 'DT'), 
            ('bank', 'NN'), 
            ('.', '.')
        ])
        self.assertIsInstance(ret, list)
        self.assertGreater(len(ret), 0)
        for i in range(len(ret)):
            self.assertIsInstance(ret[i], str)
    



