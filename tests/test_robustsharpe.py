from robustsharpe import __version__
from robustsharpe import relative_hac_inference
import unittest
import numpy as np


def test_version():
    assert __version__ == '0.1.0'


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.ret_agg = np.load("../data/ret_agg.npy")
        self.ret_hedge = np.load("../data/ret_hedge.npy")

    def test_relative_sharpe(self):
        expected_se = 0.052581574094722326
        res = relative_hac_inference(self.ret_agg, rf=0.0)
        self.assertAlmostEqual(expected_se, res[-1])
