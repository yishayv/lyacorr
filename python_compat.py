import sys

if sys.version_info < (3,):
    # noinspection PyUnresolvedReferences
    from itertools import imap as map

    # noinspection PyShadowingBuiltins
    range = xrange
else:
    # noinspection PyUnresolvedReferences
    from functools import reduce
