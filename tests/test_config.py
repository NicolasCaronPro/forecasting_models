import unittest
from typing import Any, Dict
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.configs.config import Config

class TestConfig(unittest.TestCase):

    def setUp(self):
        self.config = Config()

    def test_get_set(self):
        self.config.set("key1", "value1")
        self.assertEqual(self.config.get("key1"), "value1")

    def test_get_nonexistent_key(self):
        self.assertIsNone(self.config.get("nonexistent_key"))

    def test_set_existing_key(self):
        self.config.set("key1", "value1")
        self.config.set("key1", "new_value")
        self.assertEqual(self.config.get("key1"), "new_value")

    def test_getitem_setitem(self):
        self.config["key1"] = "value1"
        self.assertEqual(self.config["key1"], "value1")

    def test_getitem_nonexistent_key(self):
        with self.assertRaises(KeyError):
            value = self.config["nonexistent_key"]

    def test_setitem_existing_key(self):
        self.config["key1"] = "value1"
        self.config["key1"] = "new_value"
        self.assertEqual(self.config["key1"], "new_value")

    def test_iter(self):
        self.config.set("key1", "value1")
        self.config.set("key2", "value2")
        keys = [key for key in self.config]
        self.assertEqual(keys, ["key1", "key2"])

    def test_len(self):
        self.config.set("key1", "value1")
        self.config.set("key2", "value2")
        self.assertEqual(len(self.config), 2)

    def test_contains(self):
        self.config.set("key1", "value1")
        self.assertTrue("key1" in self.config)
        self.assertFalse("nonexistent_key" in self.config)

    def test_keys(self):
        self.config.set("key1", "value1")
        self.config.set("key2", "value2")
        keys = self.config.keys()
        self.assertEqual(list(keys), ["key1", "key2"])

    # def test_values(self):
    #     self.config.set("key1", "value1")
    #     self.config.set("key2", "value2")
    #     values = self.config.values()
    #     self.assertEqual(list(values), ["value1", "value2"])

    def test_items(self):
        self.config.set("key1", "value1")
        self.config.set("key2", "value2")
        items = self.config.items()
        self.assertEqual(list(items), [("key1", "value1"), ("key2", "value2")])

    def test_update(self):
        self.config.set("key0", "value0")
        self.config.set("key1", "value1")
        config_dict = {"key0": "value0", "key1": "new_value1", "key2": "value2", "key3": "value3"}
        self.config.update(config_dict)
        self.assertEqual(self.config.to_dict(), config_dict)

    def test_to_dict(self):
        self.config.set("key1", "value1")
        self.config.set("key2", "value2")
        config_dict = self.config.to_dict()
        self.assertEqual(config_dict, {"key1": "value1", "key2": "value2"})

    def test_from_dict(self):
        config_dict = {"key1": "value1", "key2": "value2"}
        self.config.from_dict(config_dict)
        self.assertEqual(self.config.to_dict(), config_dict)

    def test_save_load(self):
        config_dict = {"key1": "value1", "key2": "value2"}
        self.config.from_dict(config_dict)
        self.config.save("test_config.pkl")
        loaded_config = Config()
        loaded_config.load("test_config.pkl")
        self.assertEqual(loaded_config.to_dict(), config_dict)

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()