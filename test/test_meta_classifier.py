import OpenAttack
import numpy as np
import unittest, os

class MetaClassifier(OpenAttack.Classifier):
    def __init__(self):
        self.last_meta = None
    
    def get_pred(self, input_):
        return self.get_prob(input_)
    
    def get_prob(self, input_):
        return self.get_grad([input_], [0])[0]
    
    def get_grad(self, input_, labels):
        self.last_meta = self.context.input
        return np.array([[1, 2, 3]]), None

class TestMetaClassifier(unittest.TestCase):
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
        clsf.set_context({}, None)
        clsf.get_pred(["I love apples"])
        self.assertDictEqual(clsf.last_meta, {})
        clsf.set_context({"THIS": "that"}, None)
        clsf.get_pred(["I love apples"])
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
        clsf.set_context({}, None)
        clsf.get_prob(["I love apples"])
        self.assertDictEqual(clsf.last_meta, {})
        clsf.set_context({"THIS": "that"}, None)
        clsf.get_prob(["I love apples"])
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
        clsf.set_context({}, None)
        clsf.get_grad([["I", "love", "apples"]], [0])
        self.assertDictEqual(clsf.last_meta, {})
        clsf.set_context({"THIS": "that"}, None)
        clsf.get_grad([["I", "love", "apples"]], [0])
        self.assertDictEqual(clsf.last_meta, {"THIS": "that"})