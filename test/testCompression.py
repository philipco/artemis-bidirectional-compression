"""
Created by Constantin Philippenko, 8th August 2021.
"""
import unittest

import torch

from src.models.CompressionModel import SQuantization


class PerformancesTest(unittest.TestCase):

    def test_quantization_without_bucket(self):
        quantization = SQuantization(level=1, norm=2)
        quantization.bucket_size = 100
        zeros = torch.zeros(10)
        vector = torch.Tensor([1,2,3,4,5,6,9,1,2,3,2])
        assert quantization.__qtzt__(zeros).equal(zeros), "Compressing a zeros vector must return zeros."
        torch.manual_seed(10)
        single_qutzt = quantization.__qtzt__(vector)
        torch.manual_seed(10)
        bucket_qtzt = quantization.compress(vector)
        assert bucket_qtzt.equal(single_qutzt), "A vector with less element than the bucket size should be quantized as a single vector."

    def test_quantization_with_bucket(self):
        quantization = SQuantization(level=1, norm=2)
        quantization.bucket_size = 4
        vector = torch.Tensor([1,2,3,4,11,2,30,4,8,1])
        bucket_quantization = torch.Tensor([0.0000, 0.0000, 0.0000, 5.4772257805, 0.0000,  0.0000, 32.2645301819, 32.2645301819, 8.0622577667, 0.0000])
        torch.manual_seed(10)
        a = quantization.compress(vector)
        print(a)
        assert a.equal(bucket_quantization), "The quantization by bucket is incorrect."

