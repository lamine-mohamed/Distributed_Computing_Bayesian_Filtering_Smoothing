import numpy as np
from mpi4py import MPI

def simultaneous_binomial_tree_scan(local_val, OP):
    """
    Inclusive scan using the Simultaneous Binomial Tree algorithm.

    Each process computes an inclusive prefix operation across all ranks 
    using a binomial tree communication pattern. In each round `k` (0 <= k < ⎡log₂ p⎤),
    every process sends its current accumulator to the rank `rank + 2^k` (if that rank exists)
    and receives from rank `rank - 2^k` (if available). After receiving the value,
    the process updates its accumulator using the provided binary operation `OP`.

    Parameters
    ----------
    local_val : object
        Local data at the current MPI process.
    OP : callable
        Binary associative operation. Must take two arguments
        and return a result of the same type.

    Returns
    -------
    y : same type as local_val
        Inclusive scan result for the current process.

    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    y = local_val
    num_rounds = int(np.ceil(np.log2(size))) if size > 1 else 0

    for k in range(num_rounds):
        offset = 2 ** k
        send_req = None

        if rank + offset < size:
            dest = rank + offset
            send_req = comm.isend(y, dest=dest, tag=200 + k)

        if rank - offset >= 0:
            src = rank - offset
            received_val = comm.recv(source=src, tag=200 + k)
            y = OP(received_val, y)

        if send_req is not None:
            send_req.wait()

    return y


from mpi4py import MPI
import numpy as np


def exclusive_simultaneous_binomial_tree_scan(input, OP, **kwargs):
    """
    Perform an exclusive scan using the Simultaneous Binomial Tree algorithm.

    In each round :math:`k` (0 ≤ k < ⌈log₂(p)⌉), every process:
      1. Sends its current partial value to process `rank + 2^k` (if that process exists).
      2. Receives a partial value from process `rank - 2^k` (if that process exists).
      3. Updates its accumulator by applying the operation `OP` to the received value.

    Parameters
    ----------
    input : any
        The local value for the current MPI process.
    OP : callable
        A binary operation with an `identity` attribute, defining the scan reduction.
        Must support the signature `OP(a, b)`.
    **kwargs : dict, optional
        Additional keyword arguments.
        - comm : MPI.Comm, optional
            MPI communicator to use. Defaults to `MPI.COMM_WORLD`.

    Returns
    -------
    result : any
        The exclusive scan result for the calling process.

    Notes
    -----
    - This implementation uses non-blocking sends (`isend`) and blocking receives (`recv`).
    - The complexity is O(log p) communication rounds.
    """
    comm = kwargs.get('comm', MPI.COMM_WORLD)
    rank = comm.Get_rank()
    size = comm.Get_size()

    y = input  # local accumulator
    result = OP.identity  # initialize with identity element
    num_rounds = int(np.ceil(np.log2(size))) if size > 1 else 0

    send_req = None
    for k in range(num_rounds):
        offset = 2 ** k

        # Send to higher-ranked neighbor
        if rank + offset < size:
            dest = rank + offset
            y = OP(result, input)
            send_req = comm.isend(y, dest=dest, tag=200 + k)

        # Receive from lower-ranked neighbor
        if rank - offset >= 0:
            src = rank - offset
            received_val = comm.recv(source=src, tag=200 + k)
            result = OP(received_val, result)

        # Ensure send completes
        if send_req is not None:
            send_req.wait()

    return result


def exclusive_simultaneous_binomial_tree_scan_v2(input, OP, **kwargs):
    """
    Perform an exclusive scan using an alternative Simultaneous Binomial Tree approach.

    This version initializes communication with immediate neighbor exchange before
    proceeding with the binomial tree rounds. The result is exclusive, meaning the
    calling process does not include its own `input` value in the final result.

    Parameters
    ----------
    input : any
        The local value for the current MPI process.
    OP : callable
        A binary operation with an `identity` attribute, defining the scan reduction.
        Must support the signature `OP(a, b)`.
    **kwargs : dict, optional
        Additional keyword arguments.
        - comm : MPI.Comm, optional
            MPI communicator to use. Defaults to `MPI.COMM_WORLD`.

    Returns
    -------
    result : any
        The exclusive scan result for the calling process.

    Notes
    -----
    - The algorithm starts by exchanging values with immediate neighbors.
    - Then it proceeds with log₂(p) binomial tree communication rounds.

    """
    comm = kwargs.get('comm', MPI.COMM_WORLD)
    rank = comm.Get_rank()
    size = comm.Get_size()

    y = input
    num_rounds = int(np.ceil(np.log2(size))) if size > 1 else 0

    send_req = None

    # Initial neighbor exchange
    if rank + 1 < size:
        send_req = comm.isend(y, dest=rank + 1, tag=100)
    if rank - 1 >= 0:
        y = comm.recv(source=rank - 1, tag=100)
    if send_req is not None:
        send_req.wait()

    if rank == 0:
        y = OP.identity

    # Binomial tree rounds
    for k in range(num_rounds):
        offset = 2 ** k

        # Send to higher-ranked neighbor
        if rank + offset < size:
            dest = rank + offset
            send_req = comm.isend(y, dest=dest, tag=200 + k)

        # Receive from lower-ranked neighbor
        if rank - offset >= 0:
            src = rank - offset
            received_val = comm.recv(source=src, tag=200 + k)
            y = OP(received_val, y)

        if send_req is not None:
            send_req.wait()

    return y
