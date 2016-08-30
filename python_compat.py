import sys

if sys.version_info < (3,):
    # noinspection PyUnresolvedReferences,PyShadowingBuiltins
    from itertools import imap as map

    # noinspection PyShadowingBuiltins
    range = xrange
    # noinspection PyShadowingBuiltins
    reduce = reduce
else:
    # noinspection PyUnresolvedReferences
    from functools import reduce
    # noinspection PyShadowingBuiltins
    range = range
