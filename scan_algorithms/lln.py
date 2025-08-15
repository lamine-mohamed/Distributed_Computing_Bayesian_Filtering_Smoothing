from mpi4py import MPI
import numpy as np
from scripts.utils import safe_copy

def fibonacci(k):
    """
    Compute the k-th Fibonacci number recursively.

    Parameters
    ----------
    k : int
        Index of the Fibonacci number (0-based).

    Returns
    -------
    int
        The k-th Fibonacci number.

    Notes
    -----
    This is a naive recursive implementation and may be slow for large k.
    Consider using memoization for performance improvement.
    """
    if k == 0:
        return 0
    elif k == 1:
        return 1
    else:
        return fibonacci(k - 1) + fibonacci(k - 2)


def compute_Lk(k):
    """
    Precompute the L_k value used in the LL(N) algorithm.

    Parameters
    ----------
    k : int
        Step index.

    Returns
    -------
    int
        Computed L_k value: L_k = F_{k+3} + F_{k+1}
    """
    return fibonacci(k + 3) + fibonacci(k + 1)


def compute_m(N):
    """
    Determine the number of stages `m` based on N.

    Parameters
    ----------
    N : int
        Half of the number of MPI processes (floor(size / 2)).

    Returns
    -------
    int
        Number of stages m such that L_m >= N.
    """
    m = 2
    while compute_Lk(m) < N:
        m += 1
    return m


def lln(local_value, OP, **kwargs):
    """
    Parallel prefix (scan) computation using the LL(N) Fibonacci-based algorithm.

    Parameters
    ----------
    local_value : object
        Local input value for the MPI process.
    OP : object
        Binary operation object with the interface `OP(a, b)` and `OP.identity`.
    comm : mpi4py.MPI.Comm, optional
        MPI communicator. Defaults to `MPI.COMM_WORLD`.

    Returns
    -------
    object
        Computed prefix value for the local rank.
    """
    comm = kwargs.get('comm', MPI.COMM_WORLD)
    rank = comm.Get_rank()
    size = comm.Get_size()

    N = size // 2
    m = compute_m(N)
    c_value = safe_copy(local_value)

    # Stage A
    if rank % 2 == 0 and rank < size - 1:
        comm.send(local_value, dest=rank + 1, tag=0)
    elif rank % 2 == 1:
        received_value = comm.recv(source=rank - 1, tag=0)
        c_value = OP(received_value, local_value)

    if rank % 2 == 1 and 1 < rank < size - 1:
        comm.send(c_value, dest=rank - 1, tag=1)
    elif rank % 2 == 0 and 0 < rank < size - 2:
        received_value = comm.recv(source=rank + 1, tag=1)
        c_value = received_value

    # Stage B (Steps 3 to m)
    for k in range(3, m + 1):
        Fk = fibonacci(k)
        Fk_2 = fibonacci(k - 2)
        if m % 2 == 0:
            if k % 2 == 1:
                if rank % 2 == 1 and rank < 2 * (N - Fk):
                    comm.send(c_value, dest=rank + 2 * Fk_2 - 1, tag=2)
                elif rank % 2 == 0 and rank >= 2 * Fk_2 and rank < 2 * (N - Fk + Fk_2):
                    received_value = comm.recv(source=rank - 2 * Fk_2 + 1, tag=2)
                    c_value = OP(received_value, c_value)
            elif k % 2 == 0:
                if rank % 2 == 0 and rank < 2 * (N - Fk_2):
                    comm.send(c_value, dest=rank + 2 * Fk_2 + 1, tag=2)
                elif rank % 2 == 1 and rank >= 2 * Fk_2 + 1 and rank < 2 * N + 1:
                    received_value = comm.recv(source=rank - 2 * Fk_2 - 1, tag=2)
                    c_value = OP(received_value, c_value)
        else:
            if k % 2 == 1:
                if rank % 2 == 0 and rank < 2 * (N - Fk_2):
                    comm.send(c_value, dest=rank + 2 * Fk_2 + 1, tag=2)
                elif rank % 2 == 1 and rank >= 2 * Fk_2 + 1 and rank < 2 * N + 1:
                    received_value = comm.recv(source=rank - 2 * Fk_2 - 1, tag=2)
                    c_value = OP(received_value, c_value)
            elif k % 2 == 0:
                if rank % 2 == 1 and rank < 2 * (N - Fk) + 1:
                    comm.send(c_value, dest=rank + 2 * Fk_2 - 1, tag=2)
                elif rank % 2 == 0 and rank >= 2 * Fk_2 and rank < 2 * (N - Fk + Fk_2):
                    received_value = comm.recv(source=rank - 2 * Fk_2 + 1, tag=2)
                    c_value = OP(received_value, c_value)

    # Stage C to F
    Fm = fibonacci(m)
    Fm_2 = fibonacci(m - 2)
    Fm1 = fibonacci(m + 1)
    Fm_1 = fibonacci(m - 1)

    # Stage C
    if rank % 2 == 1 and rank < 2 * Fm + 1:
        comm.send(c_value, dest=rank + 2 * Fm, tag=3)
    if rank % 2 == 0 and rank >= 2 * Fm and rank < 2 * (N - Fm):
        comm.send(c_value, dest=rank + 2 * Fm + 1, tag=3)
    if rank % 2 == 1 and rank - 2 * Fm >= 0:
        received_value = comm.recv(source=rank - 2 * Fm - int((rank / 2 - 2 * Fm) >= 0), tag=3)
        c_value = OP(received_value, c_value)

    # Stage D
    if rank >= 2 * Fm_2 + 1 and rank < 2 * (Fm1 + Fm_2) and rank % 2 == 1:
        comm.send(c_value, dest=rank - 2 * Fm_2 + 4 * Fm, tag=4)
    if rank >= 4 * Fm + 1 and rank < 2 * (Fm1 + 2 * Fm) and rank % 2 == 1:
        received_value = comm.recv(source=rank + 2 * Fm_2 - 4 * Fm, tag=4)
        c_value = OP(received_value, c_value)
    if rank >= 4 * Fm and rank < 2 * (N - Fm1) and rank % 2 == 0:
        comm.send(c_value, dest=rank + 2 * Fm1 + 1, tag=4)
    if rank >= 2 * (2 * Fm + Fm1) + 1 and rank < 2 * N + 1 and rank % 2 == 1:
        received_value = comm.recv(source=rank - 2 * Fm1 - 1, tag=4)
        c_value = OP(received_value, c_value)

    # Stage E
    if rank >= 2 * (2 * Fm - Fm_1) + 1 and rank < 2 * (N - Fm_1 - Fm1) + 1 and rank % 2 == 1:
        comm.send(c_value, dest=rank + 2 * (Fm_1 + Fm1), tag=5)
    if rank >= 2 * (2 * Fm + Fm1) + 1 and rank < 2 * N + 1 and rank % 2 == 1:
        received_value = comm.recv(source=rank - 2 * (Fm_1 + Fm1), tag=5)
        c_value = OP(received_value, c_value)

    # Stage F
    if rank % 2 == 1 and rank < size - 1:
        comm.send(c_value, dest=rank + 1, tag=6)
    if rank % 2 == 0 and rank > 0:
        received_value = comm.recv(source=rank - 1, tag=6)
        c_value = OP(received_value, local_value)

    return c_value


def exclusive_LLN(input, OP, **kwargs):
    """
    Compute an exclusive prefix scan using LL(N) algorithm.

    Parameters
    ----------
    input : object
        Input value for the current MPI rank.
    OP : object
        Binary operation object with `OP(a, b)` and `OP.identity`.
    comm : mpi4py.MPI.Comm, optional
        MPI communicator. Defaults to `MPI.COMM_WORLD`.

    Returns
    -------
    object
        Exclusive prefix value for the local rank.
    """
    comm = kwargs.get('comm', MPI.COMM_WORLD)
    rank = comm.Get_rank()
    size = comm.Get_size()

    send_req = None
    if rank + 1 < size:
        send_req = comm.isend(input, dest=rank + 1, tag=100)

    if rank - 1 >= 0:
        input = comm.recv(source=rank - 1, tag=100)

    if send_req is not None:
        send_req.wait()

    if rank == 0:
        input = OP.identity

    return lln(input, OP, comm=comm)
