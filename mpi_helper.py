from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD


def r_print(*args):
    if comm.rank == 0:
        print 'ROOT:',
        for i in args:
            print i,
        print


def l_print(*args):
    for rank in range(0, comm.size):
        comm.Barrier()
        if rank == comm.rank:
            print comm.rank, ':',
            for i in args:
                print i,
            print
        comm.Barrier()


# divide items into n=num_steps chunks
def get_chunks(num_items, num_steps):
    chunk_sizes = np.zeros(num_steps, dtype=int)
    chunk_sizes[:] = num_items // num_steps
    chunk_sizes[:num_items % num_steps] += 1

    chunk_offsets = np.roll(np.cumsum(chunk_sizes), 1)
    chunk_offsets[0] = 0
    return chunk_sizes, chunk_offsets