import unittest
import TAADToolbox as tat
import os
import pickle


class TestDataManager(unittest.TestCase):
    def setUp(self):
        os.system("mkdir ./testdir")
        tat.DataManager.set_path("./testdir")
        tat.DataManager.setAutoDownload(False)

    def tearDown(self):
        os.system("rm -r ./testdir")

    def test_load(self):
        with self.assertRaises(tat.exceptions.UnknownDataException):
            tat.DataManager.load("unknown_data")
        with self.assertRaises(tat.exceptions.DataNotExistException):
            tat.DataManager.load("test")
        tat.DataManager.download("test")
        self.assertDictEqual(tat.DataManager.load("test"), {"result": "success"})

    def test_get(self):
        with self.assertRaises(tat.exceptions.UnknownDataException):
            tat.DataManager.get("unknown_data")
        self.assertEqual("./testdir/test", tat.DataManager.get("test"))
        tat.DataManager.set_path("/home")
        self.assertEqual("/home/test", tat.DataManager.get("test"))
        tat.DataManager.set_path("/home/123", "test")
        self.assertEqual("/home/123", tat.DataManager.get("test"))
        tat.DataManager.set_path("/home/test", "test")

    def test_set_path(self):
        with self.assertRaises(tat.exceptions.UnknownDataException):
            tat.DataManager.set_path("/data", "unknown_data")
