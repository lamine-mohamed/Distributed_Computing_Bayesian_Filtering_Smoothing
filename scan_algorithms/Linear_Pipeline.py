import sys
from mpi4py import MPI
import numpy as np

def linear_pipeline_scan(local_data, op, **kwargs):
    """
    Perform a **linear pipeline parallel prefix sum** over distributed blocks of data.

    Each process divides its local array into blocks and pipelines them through
    neighboring MPI ranks. Blocks are updated sequentially with contributions
    from the previous rank to compute the global prefix sum.

    Parameters
    ----------
    local_data : np.ndarray
        Local 1D array of elements on this MPI process.
    op : callable
        Associative binary operation to apply (e.g., addition).
    b : int, optional
        Number of blocks to divide `local_data` into. If not provided,
        defaults to sqrt(m * log2(size)), adjusted to divide `local_data` evenly.

    Returns
    -------
    np.ndarray
        Local array updated with prefix sums including contributions from
        lower-rank processes.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    m = len(local_data)
    if m == 0:
        return local_data

    # Determine block count
    b = kwargs.get('b', None)
    if b is None:
        n = int(np.ceil(np.log2(size+1))) - 1 if size > 1 else 0
        b = int(np.sqrt(m * n)) if size > 1 else 1
    while m % b != 0:
        b -= 1

    # Split local data into blocks
    blocks = np.array_split(local_data, b)
    new_blocks = np.empty_like(blocks)

    if rank == 0:
        for r in range(b):
            comm.Send(blocks[r], dest=rank+1, tag=r)
    elif rank == size - 1:
        recv = np.empty_like(blocks[0])
        comm.Recv(recv, source=rank-1, tag=0)
        for r in range(1, b):
            recv_block = np.empty_like(blocks[r])
            req = comm.Irecv(recv_block, source=rank-1, tag=r)
            updated_block = op(recv, blocks[r-1])
            new_blocks[r-1] = updated_block
            recv = recv_block
            req.wait()
        updated_block = op(recv, blocks[b-1])
        new_blocks[b-1] = updated_block
    else:
        for r in range(b):
            if r > 0:
                req = comm.Isend(new_blocks[r-1], dest=rank+1, tag=r-1)
            recv_block = np.empty_like(blocks[r])
            comm.Recv(recv_block, source=rank-1, tag=r)
            updated_block = op(recv_block, blocks[r])
            new_blocks[r] = updated_block
            if r > 0:
                req.wait()
        comm.Send(new_blocks[b-1], dest=rank+1, tag=b-1)

    return np.concatenate(blocks if rank == 0 else new_blocks)


def exclusive_linear_scan(local_data, op, **kwargs):
    """
    Perform an **exclusive linear pipeline prefix sum** over MPI processes.

    Each process receives the accumulated result from the previous rank and
    updates its own local data without including its own contribution in
    the returned result.

    Parameters
    ----------
    local_data : np.ndarray
        Local 1D array of elements on this MPI process.
    op : callable
        Associative binary operation to apply (must have `.identity` attribute).
    comm : MPI.Comm, optional
        Custom MPI communicator (defaults to `MPI.COMM_WORLD`).

    Returns
    -------
    np.ndarray
        Exclusive prefix sum from all lower-rank processes, excluding
        contributions from this rank.
    """
    comm = kwargs.get('comm', MPI.COMM_WORLD)
    rank = comm.Get_rank()
    size = comm.Get_size()

    result = op.identity
    if rank == 0:
        comm.send(local_data, dest=1, tag=0)
    elif rank == size - 1:
        result = comm.recv(source=rank - 1, tag=0)
    else:
        result = comm.recv(source=rank - 1, tag=0)
        local_data = op(result, local_data)
        comm.send(local_data, dest=rank + 1, tag=0)
    return result
