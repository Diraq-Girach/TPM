import abc
import math
import os
import random
from sympy.combinatorics.graycode import GrayCode
import numpy as np

from coding import Code


class AbstractWeightsGenerator:
    def __init__(self, k, n, l, *args, **kwargs):
        pass

    def __call__(self):
        raise NotImplementedError


class WeightsGenerator(AbstractWeightsGenerator):
    def __init__(self, k, n, l):
        super().__init__(k, n, l)
        self.l = l
        self.n = n
        self.k = k

    def __call__(self):
        return np.random.randint(-self.l, self.l + 1, size=(self.k, self.n))


class ErrorChannelWeightGenerator(AbstractWeightsGenerator):
    def __init__(self, k, n, l, code=None, error_prob=0.1):
        super().__init__(k, n, l)
        assert code

        self.bits_per_w = int(math.log2(l))
        self.k = k
        self.l = l
        self.n = n
        self.calls = 0
        bit_len = k * n * self.bits_per_w

        self._rand = random.getrandbits(bit_len)
        self._rand_copy = None
        self._rand_mask = self._get_rand_mask(bit_len, error_prob)
        self._code = code(self.bits_per_w, 2 ** (self.bits_per_w - 1) - 1)

    def __call__(self):
        self.calls += 1
        assert self.calls <= 2, "Error channel weight generator cannot produce weights more than twice"

        if self.calls == 1:
            self._rand_copy = self._rand
        else:
            self._rand_copy = self._rand ^ self._rand_mask
        # generate rands from self._random
        rands = []
        for _ in range(self.k):
            rands_inner = [self.get_value(self.bits_per_w) for _ in range(self.n)]
            rands.append(rands_inner)
        # convert it to numpy array
        return np.array(rands)

    def get_value(self, n_bits):
        # get last n_bits bits
        val = self._rand_copy & (2 ** n_bits - 1)
        # remove used bits from _rand
        self._rand_copy >>= n_bits
        return self._code.decode(val)

    @staticmethod
    def _get_rand_mask(bit_len, error_prob):
        val = 0
        for _ in range(int(bit_len)):
            # Correctly shift left by 1, then use bitwise OR to set the last bit
            bit = 1 if random.random() < error_prob else 0
            val = (val << 1) | bit
        return val  # Changed from returning []
