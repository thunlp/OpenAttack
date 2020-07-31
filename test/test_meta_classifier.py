import OpenAttack
import numpy as np
import unittest, os

class MetaClassifier(OpenAttack.Classifier):
    def __init__(self):
        self.last_meta = None
    def get_grad(self, input_, labels, meta):
        self.last_meta = meta
        return np.array([[1, 2, 3]]), None

class TestMetaClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OpenAttack.DataManager.set_path("./testdir")
        OpenAttack.DataManager.download("TProcess.NLTKSentTokenizer")
        OpenAttack.DataManager.download("TProcess.NLTKPerceptronPosTagger")
    
    @classmethod
    def tearDownClass(cls):
        os.system("rm -r ./testdir")

    def test_get_pred(self):
        clsf = MetaClassifier()
        self.assertIsNone(clsf.last_meta)
        with self.assertRaises(TypeError):
            clsf.get_pred("I love apples")
        with self.assertRaises(TypeError):
            clsf.get_pred()
        with self.assertRaises(TypeError):
            clsf.get_pred(["I love apples"], "b", "c")
        self.assertIsNone(clsf.last_meta)
        clsf.get_pred(["I love apples"])
        self.assertDictEqual(clsf.last_meta, {})
        clsf.get_pred(["I love apples"], {"THIS": "that"})
        self.assertDictEqual(clsf.last_meta, {"THIS": "that"})

    def test_get_prob(self):
        clsf = MetaClassifier()
        self.assertIsNone(clsf.last_meta)
        with self.assertRaises(TypeError):
            clsf.get_prob("I love apples")
        with self.assertRaises(TypeError):
            clsf.get_prob()
        with self.assertRaises(TypeError):
            clsf.get_prob(["I love apples"], "b", "c")
        self.assertIsNone(clsf.last_meta)
        clsf.get_prob(["I love apples"])
        self.assertDictEqual(clsf.last_meta, {})
        clsf.get_prob(["I love apples"], {"THIS": "that"})
        self.assertDictEqual(clsf.last_meta, {"THIS": "that"})

    def test_get_grad(self):
        clsf = MetaClassifier()
        self.assertIsNone(clsf.last_meta)
        with self.assertRaises(TypeError):
            clsf.get_grad("I love apples")
        with self.assertRaises(TypeError):
            clsf.get_grad()
        with self.assertRaises(TypeError):
            clsf.get_grad(["I love apples"])
        with self.assertRaises(TypeError):
            clsf.get_grad(["I love apples"], "b", "c", "d")
        self.assertIsNone(clsf.last_meta)
        clsf.get_grad(["I love apples"], [0])
        self.assertDictEqual(clsf.last_meta, {})
        clsf.get_grad(["I love apples"], [0], {"THIS": "that"})
        self.assertDictEqual(clsf.last_meta, {"THIS": "that"})