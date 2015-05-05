import numpy as np

__author__ = 'yishay'


def reverse_dict(d):
    return dict((v, k) for k, v in d.iteritems())


class PixelFlags:
    def __init__(self):
        pass

    FlagNames = \
        {0: 'NOPLUG', 1: 'BADTRACE', 2: 'BADFLAT', 3: 'BADARC',
         4: 'MANYBADCOLUMNS', 5: 'MANYREJECTED', 6: 'LARGESHIFT', 7: 'BADSKYFIBER',
         8: 'NEARWHOPPER', 9: 'WHOPPER', 10: 'SMEARIMAGE', 11: 'SMEARHIGHSN',
         12: 'SMEARMEDSN', 13: 'UNUSED_13', 14: 'UNUSED_14', 15: 'UNUSED_15',
         16: 'NEARBADPIXEL', 17: 'LOWFLAT', 18: 'FULLREJECT', 19: 'PARTIALREJECT',
         20: 'SCATTEREDLIGHT', 21: 'CROSSTALK', 22: 'NOSKY', 23: 'BRIGHTSKY',
         24: 'NODATA', 25: 'COMBINEREJ', 26: 'BADFLUXFACTOR', 27: 'BADSKYCHI',
         28: 'REDMONSTER', 29: 'UNUSED_29', 30: 'UNUSED_30', 31: 'UNUSED_31'}
    FlagValues = reverse_dict(FlagNames)

    @classmethod
    def string_to_int(cls, bit_string):
        str_list = bit_string.split('|')
        bit_list = [1 << cls.FlagValues[i] for i in str_list]
        return np.array(bit_list).sum()

    @classmethod
    def int_to_string(cls, flags):
        bit_string = ''
        for i in xrange(32):
            if flags & 1:
                if bit_string:
                    bit_string += '|'
                bit_string += (cls.FlagNames[i])
            flags >>= 1
        return bit_string


class FlagStats:
    def __init__(self):
        self.flag_count = np.zeros(shape=(32, 2), dtype=np.uint64)
        self.pixel_count = np.uint64()

    def bit_fraction(self, bit, and_or):
        return self.flag_count[bit, and_or] / self.pixel_count

    def to_string(self, bit):
        return '{bit_number:4}: {bit_name:24}: AND:{and_fraction:8.2%} OR:{or_fraction:8.2%}'.format(
            bit_number=bit, bit_name=self.FlagNames[bit],
            and_fraction=self.bit_fraction(bit, 0),
            or_fraction=self.bit_fraction(bit, 1))


