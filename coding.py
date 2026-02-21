from sympy.combinatorics.graycode import GrayCode as s_GrayCode

from tools import SetterProperty


class Code:
    def __init__(self):
        self._code = None
        self._inverse_code = None

    @SetterProperty
    def code(self, value):
        self._code = value
        self._inverse_code = self._inverse_dict(value)

    @staticmethod
    def _inverse_dict(dct: dict):
        return {v: k for k, v in dct.items()}

    def encode(self, val):
        return self._code[val]

    def decode(self, val):
        return self._inverse_code[val]


class GrayCode(Code):
    def __init__(self, length, offset):
        super().__init__()
        self.length = length
        self.offset = offset
        self.code = self._get_gray_dict(s_GrayCode(length), offset)

    @staticmethod
    def _get_gray_dict(code, offset):
        return {k - offset: GrayCode._bool_string_to_int(v) for k, v in enumerate(list(code.generate_gray()))}

    @staticmethod
    def _bool_string_to_int(string):
        val = 0
        for char in string:
            val <<= 1
            val += int(char)
        return val


class TwoComplimentCode(Code):
    def __init__(self, length, offset):
        super().__init__()
        self.offset = offset
        self.length = length
        self.code = self._get_twos_complement_dict(length, offset)

    @staticmethod
    def _get_twos_complement_dict(length, offset):
        return {k - offset: k for k in range(2**length)}

    @staticmethod
    def _twos_complement(value, bits):
        if value < 0:
            value = (1 << bits) + value
        formatstring = '{:0%ib}' % bits
        return formatstring.format(value)
