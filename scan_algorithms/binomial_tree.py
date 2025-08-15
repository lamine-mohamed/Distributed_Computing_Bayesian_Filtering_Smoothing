import sys
import numpy as np
from mpi4py import MPI

def binomial_tree_scan(local_data, OP):
    """
    Perform an **inclusive scan** (parallel prefix) using the **binomial tree algorithm** in MPI.

    This implementation computes the cumulative result of applying a binary associative 
    operation (`OP`) to all ranks up to and including the current rank.  
    Each process ends up with the scan result for all ranks from `0` to `rank`.

    Algorithm Overview:
    -------------------
    - **Up Phase:** Partial results are propagated toward higher-ranked processes 
      following a binomial tree pattern. At each round `k`, processes send
      data to a partner at distance `2^k`.
    - **Down Phase:** After partial reductions, the computed prefix sums are propagated 
      back down the tree so that every process gets its correct inclusive scan result.

    Parameters
    ----------
    local_data : any
        The data local to this MPI rank. Must be compatible with `OP`.
    OP : callable
        A binary associative function taking two arguments and returning the result 
        (e.g., `lambda a, b: a + b`).
    Returns
    -------
    y : same type as local_data
        The inclusive scan result for this process.

    Notes
    -----
    - Requires that `OP` is associative.
    - Communicator used is `MPI.COMM_WORLD`.
    - Designed for small to moderate message sizes; no explicit optimization for 
      very large data volumes.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    y = local_data
    n = int(np.ceil(np.log2(size))) if size > 1 else 0

    try:
        # --- Up Phase ---
        for k in range(n):
            pew = 2**k
            paw = 2**(k + 1)

            if rank >= pew and (rank % paw == paw - 1):
                src = rank - pew
                received_val = comm.recv(source=src, tag=k)
                y = OP(received_val, y)

            elif (rank + pew < size) and ((rank + pew) % paw == paw - 1):
                dest = rank + pew
                comm.send(y, dest=dest, tag=k)

        # --- Down Phase ---
        for k in range(n, 0, -1):
            pew = 2**(k - 1)
            paw = 2**k

            if (rank % paw == paw - 1) and (rank + pew < size):
                comm.send(y, dest=rank + pew, tag=100 + k)

            if rank >= pew and ((rank - pew) % paw == paw - 1):
                src = rank - pew
                received_val = comm.recv(source=src, tag=100 + k)
                y = OP(received_val, y)

    except Exception as e:
        if rank == 0:
            print(f"Rank {rank} encountered an error: {e}, line {sys.exc_info()[-1].tb_lineno}")

    return y


def binomial_tree_exclusive_scan(input, OP, **kwargs):
    """
    Perform an **exclusive scan** (parallel prefix) using the **binomial tree algorithm** in MPI.

    This implementation computes the cumulative result of applying a binary associative 
    operation (`OP`) to all ranks **before** the current rank.  
    Each process ends up with the scan result for ranks `0` to `rank-1` (or the identity 
    element if `rank == 0`).

    Algorithm Overview:
    -------------------
    - Initializes with `OP.identity` for rank 0.
    - Uses an **up phase** to compute partial prefix sums along a binomial tree pattern.
    - Uses a **down phase** to propagate correct exclusive results to all ranks.

    Parameters
    ----------
    input : any
        The value local to this MPI rank. Must be compatible with `OP`.
    OP : callable
        A binary associative function taking two arguments and returning the result 
        (e.g., `lambda a, b: a + b`). It should have an `identity` attribute that 
        defines the neutral element (e.g., `0` for addition).
    **kwargs : dict, optional
        - comm : MPI.Comm
            An MPI communicator to use instead of `MPI.COMM_WORLD`.

    Returns
    -------
    y : same type as input
        The exclusive scan result for this process.

    Notes
    -----
    - Requires that `OP` is associative and has an `identity` attribute.
    - Default communicator is `MPI.COMM_WORLD`.

    """
    if kwargs.get('comm') is not None:
        # Use the provided MPI communicator
        comm = kwargs['comm']
    else:
        # Default to the global MPI communicator       
        comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    y = input  # Local accumulator
    result = OP.identity  # Start with the identity element for exclusive scan
    n = int(np.ceil(np.log2(size))) if size > 1 else 0  # Rounds of communication
    try:
        # --- Up Phase ---
        for k in range(n):
            pew = 2**k
            paw = 2**(k + 1)
            if rank >= pew and (rank % paw == paw - 1):
                # Receive from lower half of this binomial group
                src = rank - pew
                received_val = comm.recv(source=src, tag=k)
                result = OP(received_val, result)
                y = OP(result, input)
            elif (rank + pew < size) and ((rank + pew) % paw == paw - 1):
                # Send to upper half of this binomial group
                comm.send(y, dest=rank + pew, tag=k)

        # --- Down Phase ---
        for k in range(n, 0, -1):
            pew = 2**(k - 1)
            paw = 2**k
            if (rank % paw == paw - 1) and (rank + pew < size):
                comm.send(y, dest=rank + pew, tag=100 + k)
            if rank >= pew and ((rank - pew) % paw == paw - 1):
                src = rank - pew
                received_val = comm.recv(source=src, tag=100 + k)
                result = OP(received_val, result)
                y = OP(result, input)
    except Exception as e:
        if rank == 0:
            print(f"Rank {rank} encountered an error: {e}, line {sys.exc_info()[-1].tb_lineno}")
    comm.Barrier()
    return result



if __name__ == "__main__":
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Example local value matrix for each process
    # Each process has a matrix of size 2x2
    # local_data = np.array([[rank + 1, rank + 2], [rank + 3, rank + 4]])
    local_data = rank + 1  # Each process has a unique value

    # Define the binary operation (e.g., addition)
    # OP = lambda x, y: x @ y
    class AddOperator:
        def __call__(self, a, b):
            return a + b

        def identityelf():
            return 0

    # Perform the scan
    result = binomial_tree_exclusive_scan(local_data, AddOperator())

    # Gather results at root process
    all_results = comm.gather(result, root=0)

    if rank == 0:
        print("Final scan results from all processes:", all_results)