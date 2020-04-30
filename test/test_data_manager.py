import unittest
import TAADToolbox as tad
import os
import pickle


class TestDataManager(unittest.TestCase):
    def setUp(self):
        os.mkdir("./testdir")
        pickle.dump({"result": "success"}, open("./testdir/example", "wb"))
        tad.DataManager.set_path("/")

    def tearDown(self):
        os.remove("./testdir/example")
        os.rmdir("./testdir")

    def test_load(self):
        with self.assertRaises(tad.exceptions.UnknownDataException):
            tad.DataManager.load("unknown_data")
        with self.assertRaises(tad.exceptions.DataNotExistException):
            tad.DataManager.load("example")
        tad.DataManager.set_path(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdir")
        )
        self.assertDictEqual(tad.DataManager.load("example"), {"result": "success"})

    def test_get(self):
        with self.assertRaises(tad.exceptions.UnknownDataException):
            tad.DataManager.get("unknown_data")
        self.assertEqual("/example", tad.DataManager.get("example"))
        tad.DataManager.set_path("/home")
        self.assertEqual("/home/example", tad.DataManager.get("example"))
        tad.DataManager.set_path("/home/123", "example")
        self.assertEqual("/home/123", tad.DataManager.get("example"))
        tad.DataManager.set_path("/home/example", "example")

    def test_set_path(self):
        with self.assertRaises(tad.exceptions.UnknownDataException):
            tad.DataManager.set_path("/data", "unknown_data")
