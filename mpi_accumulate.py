import itertools
from collections import Iterable

import numpy as np
from astropy import table as table
from mpi4py import MPI

import common_settings
import mpi_helper
from mpi_helper import l_print_no_barrier

settings = common_settings.Settings()

comm = MPI.COMM_WORLD


def split_seq(size, iterable):
    it = iter(iterable)
    item = list(itertools.islice(it, size))
    while item:
        yield item
        item = list(itertools.islice(it, size))


def accumulate_over_spectra(func, accumulator):
    qso_record_table = table.Table(np.load(settings.get_qso_metadata_npy()))  # type: table
    qso_record_count = len(qso_record_table)

    chunk_sizes, chunk_offsets = mpi_helper.get_chunks(qso_record_count, comm.size)

    local_start_index = chunk_offsets[comm.rank]
    local_size = chunk_sizes[comm.rank]
    local_end_index = local_start_index + local_size
    if comm.rank == 0:
        global_acc = accumulator(qso_record_count)

    local_qso_record_table = itertools.islice(qso_record_table, int(local_start_index),
                                              int(local_end_index))  # type: Iterable(table.Row)
    l_print_no_barrier("-----", qso_record_count, local_start_index, local_end_index, local_size)
    slice_size = settings.get_file_chunk_size()
    qso_chunks_iterable = enumerate(split_seq(slice_size, local_qso_record_table))
    for slice_number, qso_record_table_chunk in qso_chunks_iterable:
        local_result = func(qso_record_table_chunk)
        # all large data is stored in an array as the first tuple element.
        ar_local_result = local_result[0]
        # generic objects (slower) can be store at the second tuple element.
        object_local_result = local_result[1]

        assert isinstance(ar_local_result, np.ndarray)
        ar_all_results = np.zeros(shape=(comm.size,) + tuple(ar_local_result.shape))
        comm.Gatherv(ar_local_result, ar_all_results, root=0)
        ar_qso_indices = np.zeros(shape=(comm.size, slice_size), dtype=int)
        # noinspection PyTypeChecker
        comm.Gatherv(np.array([x['index'] for x in qso_record_table_chunk]), ar_qso_indices)

        # metadata, or anything else that is small, but may have complex data types is transferred as objects:
        object_all_results = comm.gather(object_local_result)

        # "reduce" results
        if comm.rank == 0:
            global_acc.accumulate(ar_all_results, ar_qso_indices, object_all_results)
            global_acc.finalize()

    l_print_no_barrier("------------------------------")
    if comm.rank == 0:
        return global_acc.return_result()
    else:
        return None, None
