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


def is_all_done(poll_time=1):
    tag = 28476237
    # send a message to all nodes
    for i in range(comm.size):
        comm.send(True, dest=i, tag=tag)

    all_done = False
    done_list = [False] * comm.size
    while not all_done:
        time.sleep(poll_time)
        # go over all nodes
        for i in range(comm.size):
            # for every node that isn't done yet,
            if not done_list[i]:
                # check for a message from this node
                if comm.Iprobe(source=i, tag=tag):
                    # consume the message, and update node status
                    done_list[i] = comm.recv(source=i, tag=tag)
                    # l_print_no_barrier(done_list)

        # update overall status
        all_done = False not in done_list
        # give the current node a chance to do something on every iteration
        yield False

    yield True
