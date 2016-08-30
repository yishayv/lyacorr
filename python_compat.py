import sys

if sys.version_info < (3,):
    # noinspection PyUnresolvedReferences,PyShadowingBuiltins
    from itertools import imap as map
    # noinspection PyUnresolvedReferences,PyShadowingBuiltins
    from itertools import izip as zip

    # noinspection PyShadowingBuiltins
    range = xrange
    # noinspection PyShadowingBuiltins
    reduce = reduce
else:
    # noinspection PyUnresolvedReferences
    from functools import reduce
    # noinspection PyShadowingBuiltins
    map = map
    # noinspection PyShadowingBuiltins
    zip = zip
    # noinspection PyShadowingBuiltins
    range = range
