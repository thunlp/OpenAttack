import unittest
import TAADToolbox as tat
import os


class TestDataManager(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tat.DataManager.set_path("./testdir")
        tat.DataManager.download("NLTKSentTokenizer")
        tat.DataManager.download("NLTKPerceptronPosTagger")
        tat.DataManager.download("NLTKWordnet")
        tat.DataManager.download("NLTKWordnetDelemma")
        tat.DataManager.download("StanfordNER")
        tat.DataManager.download("StanfordParser")
        cls.dp = tat.text_processors.DefaultTextProcessor()
    
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
    



