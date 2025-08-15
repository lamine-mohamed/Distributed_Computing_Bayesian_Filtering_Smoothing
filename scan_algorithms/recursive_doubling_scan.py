import copy
from mpi4py import MPI
import numpy as np
import time

def recursive_doubling_scan(sendbuf, op):
    """
    Perform a parallel prefix (scan) operation using the recursive doubling algorithm 
    across MPI processes.

    This function computes an inclusive scan (prefix reduction) of `sendbuf` across 
    all MPI processes in `MPI.COMM_WORLD` using a binary tree communication pattern. 
    The provided binary operation `op` is applied recursively to combine elements. 

    Parameters
    ----------
    sendbuf : object
        Input object to be reduced across MPI ranks. Each process provides 
        its own `sendbuf`.
    op : callable
        Binary operation to apply. Must take two arguments of the same 
        type as `sendbuf` and return a value of the same type.

    Returns
    -------
    recvbuf : object
        The result of the inclusive scan on the current rank. 

    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Create deep copies of the input buffer to avoid in-place modifications
    partial = copy.deepcopy(sendbuf)
    recvbuf = copy.deepcopy(sendbuf)

    mask = 1
    while mask < size:
        partner = rank ^ mask
        if partner < size:
            # Exchange partial results with partner
            tempbuf = comm.sendrecv(partial, dest=partner, sendtag=0,
                                    source=partner, recvtag=0)
            if rank > partner:
                # Higher rank combines received buffer with local partial and receive buffers
                partial = op(tempbuf, partial)
                recvbuf = op(tempbuf, recvbuf)
            else:
                # Lower rank updates partial buffer only
                partial = op(tempbuf, partial)
        mask <<= 1  # Move to next distance in the doubling pattern

    return recvbuf
