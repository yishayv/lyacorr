from unittest import TestCase

from pixel_flags import PixelFlags


__author__ = 'yishay'


class TestPixelFlags(TestCase):
    def test_string_to_int(self):
        pf = PixelFlags()
        flags = pf.string_to_int(bit_string='NOSKY|REDMONSTER')
        assert flags == 1 << 22 | 1 << 28

    def test_int_to_string(self):
        pf = PixelFlags()
        flag_string = pf.int_to_string(flags=1 << 22 | 1 << 28)
        assert flag_string == 'NOSKY|REDMONSTER'