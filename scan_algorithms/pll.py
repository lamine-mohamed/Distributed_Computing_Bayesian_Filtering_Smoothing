import sys
import numpy as np
from mpi4py import MPI

# --- Fibonacci-related utilities ---

def fibonacci(k: int) -> int:
    """
    Compute the k-th Fibonacci number recursively.

    Parameters
    ----------
    k : int
        Index of the Fibonacci sequence (0-based).

    Returns
    -------
    int
        The k-th Fibonacci number.
    """
    if k == 0:
        return 0
    elif k == 1:
        return 1
    return fibonacci(k - 1) + fibonacci(k - 2)


def compute_Lk(k: int) -> int:
    """
    Compute L_k value used in determining step bounds for PLL scan.

    Parameters
    ----------
    k : int
        Stage index.

    Returns
    -------
    int
        Computed L_k value.
    """
    return fibonacci(k + 3) + fibonacci(k + 1)


def compute_m(N: int) -> int:
    """
    Determine minimum stage 'm' such that L_k >= N.

    Parameters
    ----------
    N : int
        Number of process pairs (half of total PEs).

    Returns
    -------
    int
        Minimum stage index m satisfying L_k >= N.
    """
    m = 2
    while compute_Lk(m) < N:
        m += 1
    return m


# --- Parallel PLL(n,p) Scan Algorithm ---

def PLL(local_arr, OP):
    """
    Perform a distributed parallel prefix scan using the PLL algorithm.

    This function implements a Fibonacci-based communication
    scheme over MPI to compute a global prefix scan of a distributed array.

    Parameters
    ----------
    local_arr : ndarray
        Local segment of the input array.
    OP : callable
        Binary associative operation (e.g., np.add, np.maximum).

    Returns
    -------
    None
        The function updates `local_arr` in-place with the prefix scan result.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if local_arr.size == 0:
        return local_arr

    P = size // 2
    m = compute_m(P)

     # === Stage A: Local scan and even-odd exchange ===
    c_value = local_arr[0].copy()
    for i in range(1, len(local_arr)):
        c_value = OP(c_value, local_arr[i])
    d_value = np.zeros_like(c_value)

    if rank % 2 == 0 and rank < 2 * P:
        comm.send(c_value, dest=rank + 1, tag=0)
    elif rank % 2 == 1 and rank < 2 * P + 1:
        received = comm.recv(source=rank - 1, tag=0)
        c_value = OP(received, c_value)
        d_value = received

    if rank % 2 == 1 and rank < 2 * P - 1:
        comm.send(c_value, dest=rank - 1, tag=1)
    elif rank % 2 == 0 and rank < 2 * (P - 1):
        received = comm.recv(source=rank + 1, tag=1)
        c_value = received

    # === Stage B: Fibonacci recursive exchanges ===
    for k in range(3, m + 1):
        Fk = fibonacci(k)
        Fk_2 = fibonacci(k - 2)

        if m % 2 == 0:
            if k % 2 == 1:
                if rank % 2 == 1 and rank < 2 * (P - Fk) + 1:
                    comm.send(c_value, dest=rank + 2 * Fk_2 - 1, tag=2)
                elif rank % 2 == 0 and 2 * Fk_2 <= rank < 2 * (P - Fk + Fk_2):
                    received = comm.recv(source=rank - 2 * Fk_2 + 1, tag=2)
                    c_value = OP(received, c_value)
            else:
                if rank % 2 == 0 and rank < 2 * (P - Fk_2):
                    comm.send(c_value, dest=rank + 2 * Fk_2 + 1, tag=2)
                elif rank % 2 == 1 and 2 * Fk_2 + 1 <= rank < 2 * P + 1:
                    received = comm.recv(source=rank - 2 * Fk_2 - 1, tag=2)
                    c_value = OP(received, c_value)
                    d_value = OP(received, d_value)
        else:
            if k % 2 == 1:
                if rank % 2 == 0 and rank < 2 * (P - Fk_2):
                    comm.send(c_value, dest=rank + 2 * Fk_2 + 1, tag=2)
                elif rank % 2 == 1 and 2 * Fk_2 + 1 <= rank < 2 * P + 1:
                    received = comm.recv(source=rank - 2 * Fk_2 - 1, tag=2)
                    c_value = OP(received, c_value)
                    d_value = OP(received, d_value)
            else:
                if rank % 2 == 1 and rank < 2 * (P - Fk) + 1:
                    comm.send(c_value, dest=rank + 2 * Fk_2 - 1, tag=2)
                elif rank % 2 == 0 and 2 * Fk_2 <= rank < 2 * (P - Fk + Fk_2):
                    received = comm.recv(source=rank - 2 * Fk_2 + 1, tag=2)
                    c_value = OP(received, c_value)

    # === Stage C: Final updates based on F_m ===
    Fm = fibonacci(m)

    if rank % 2 == 1 and rank < 2 * Fm + 1:
        comm.send(c_value, dest=rank + 2 * Fm, tag=3)
    if rank % 2 == 0 and 2 * Fm <= rank < 2 * (P - Fm):
        comm.send(c_value, dest=rank + 2 * Fm + 1, tag=3)
    if rank % 2 == 1 and rank - 2 * Fm >= 0:
        received = comm.recv(source=rank - 2 * Fm - ((rank / 2 - 2 * Fm) >= 0), tag=3)
        c_value = OP(received, c_value)
        d_value = OP(received, d_value)

    # === Stage D: Recursive backward updates ===
    Fm_2 = fibonacci(m - 2)
    Fm1 = fibonacci(m + 1)

    if 2 * Fm_2 + 1 <= rank < 2 * (Fm1 + Fm_2) and rank % 2 == 1:
        comm.send(c_value, dest=rank - 2 * Fm_2 + 4 * Fm, tag=4)
    if 4 * Fm + 1 <= rank < 2 * (Fm1 + 2 * Fm) and rank % 2 == 1:
        received = comm.recv(source=rank + 2 * Fm_2 - 4 * Fm, tag=4)
        c_value = OP(received, c_value)
        d_value = OP(received, d_value)
    if 4 * Fm <= rank < 2 * (P - Fm1) and rank % 2 == 0:
        comm.send(c_value, dest=rank + 2 * Fm1 + 1, tag=4)
    if 2 * (2 * Fm + Fm1) + 1 <= rank < 2 * P + 1 and rank % 2 == 1:
        received = comm.recv(source=rank - 2 * Fm1 - 1, tag=4)
        c_value = OP(received, c_value)
        d_value = OP(received, d_value)

    # === Stage E: Final propagation to remaining nodes ===
    Fm_1 = fibonacci(m - 1)

    if 2 * (2 * Fm - Fm_1) + 1 <= rank < 2 * (P - Fm_1 - Fm1) + 1 and rank % 2 == 1:
        comm.send(c_value, dest=rank + 2 * (Fm_1 + Fm1), tag=5)
    if 2 * (2 * Fm + Fm1) + 1 <= rank < 2 * P + 1 and rank % 2 == 1:
        received = comm.recv(source=rank - 2 * (Fm_1 + Fm1), tag=5)
        c_value = OP(received, c_value)
        d_value = OP(received, d_value)

    # === Stage F: Final even-to-odd fixup and local scan ===
    if rank % 2 == 1 and rank < size - 1:
        comm.send(c_value, dest=rank + 1, tag=6)
    if rank % 2 == 0 and rank > 0:
        received = comm.recv(source=rank - 1, tag=6)
        d_value = received

    # Final local prefix update
    if rank != 0:
        local_arr[0] = OP(d_value, local_arr[0])
    for j in range(1, len(local_arr)):
        local_arr[j] = OP(local_arr[j - 1], local_arr[j])

    comm.Barrier()
    if rank == 0:
        print("Scan complete.")
        sys.stdout.flush()