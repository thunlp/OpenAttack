import unittest
import OpenAttack
import os
import pickle


class TestDataManager(unittest.TestCase):
    def setUp(self):
        os.system("mkdir ./testdir")
        OpenAttack.DataManager.set_path("./testdir")
        OpenAttack.DataManager.setAutoDownload(False)

    def tearDown(self):
        os.system("rm -r ./testdir")

    def test_load(self):
        with self.assertRaises(OpenAttack.exceptions.UnknownDataException):
            OpenAttack.DataManager.load("unknown_data")
        with self.assertRaises(OpenAttack.exceptions.DataNotExistException):
            OpenAttack.DataManager.load("test")
        OpenAttack.DataManager.download("test")
        self.assertDictEqual(OpenAttack.DataManager.load("test"), {"result": "success"})

    def test_get(self):
        with self.assertRaises(OpenAttack.exceptions.UnknownDataException):
            OpenAttack.DataManager.get("unknown_data")
        self.assertEqual("./testdir/test", OpenAttack.DataManager.get("test"))
        OpenAttack.DataManager.set_path("/home")
        self.assertEqual("/home/test", OpenAttack.DataManager.get("test"))
        OpenAttack.DataManager.set_path("/home/123", "test")
        self.assertEqual("/home/123", OpenAttack.DataManager.get("test"))
        OpenAttack.DataManager.set_path("/home/test", "test")

    def test_set_path(self):
        with self.assertRaises(OpenAttack.exceptions.UnknownDataException):
            OpenAttack.DataManager.set_path("/data", "unknown_data")
