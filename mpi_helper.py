from __future__ import print_function

import time

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
        # noinspection PyArgumentList
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
    # noinspection PyArgumentList
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


def barrier_sleep(mpi_comm=comm, tag=1747362612, sleep=0.1, use_yield=False):
    """
    As suggested by Lisandro Dalcin at:
    https://groups.google.com/forum/?fromgroups=#!topic/mpi4py/nArVuMXyyZI
    """
    size = mpi_comm.Get_size()
    if size == 1:
        return
    rank = mpi_comm.Get_rank()
    mask = 1
    while mask < size:
        dst = (rank + mask) % size
        src = (rank - mask + size) % size
        req = mpi_comm.isend(None, dst, tag)
        while not mpi_comm.Iprobe(src, tag):
            if use_yield:
                yield False
            time.sleep(sleep)
        mpi_comm.recv(None, src, tag)
        req.Wait()
        mask <<= 1
    if use_yield:
        yield True
