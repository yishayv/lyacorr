from __future__ import print_function

import numpy as np
from mpi4py import MPI

from python_compat import range

comm = MPI.COMM_WORLD


def r_print(*args):
    """
    print message on the root node (rank 0)
    :param args:
    :return:
    """
    if comm.rank == 0:
        print('ROOT:', end=' ')
        for i in args:
            print(i, end=' ')
        print()


def l_print(*args):
    """
    print message on each node, synchronized
    :param args:
    :return:
    """
    for rank in range(0, comm.size):
        comm.Barrier()
        if rank == comm.rank:
            l_print_no_barrier(*args)
        comm.Barrier()


def l_print_no_barrier(*args):
    """
    print message on each node
    :param args:
    :return:
    """
    print(comm.rank, ':', end=' ')
    for i in args:
        print(i, end=' ')
    print()


def get_chunks(num_items, num_steps):
    """
    divide items into n=num_steps chunks
    :param num_items:
    :param num_steps:
    :return: chunk sizes, chunk offsets
    """
    chunk_sizes = np.zeros(num_steps, dtype=int)
    chunk_sizes[:] = num_items // num_steps
    chunk_sizes[:num_items % num_steps] += 1

    chunk_offsets = np.roll(np.cumsum(chunk_sizes), 1)
    chunk_offsets[0] = 0
    return chunk_sizes, chunk_offsets
