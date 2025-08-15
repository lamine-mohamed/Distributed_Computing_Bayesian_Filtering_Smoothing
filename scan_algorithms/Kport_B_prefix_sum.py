from mpi4py import MPI
import numpy as np
import math

def Kport_B_prefix_sum(local_data, k=2, op=lambda x, y: x + y):
    """
    Perform a **k-port block-based parallel prefix sum** across MPI processes.

    This algorithm extends the k-port prefix sum to operate on **local blocks** of
    data on each process. Each process first computes a local prefix over its block,
    then communicates partial sums with up to `k` neighbors at each step to compute
    the global prefix.

    Parameters
    ----------
    local_data : np.ndarray
        1D array of elements on this MPI process.
    k : int, optional
        Fan-out: number of processes each process sends to at each step (default is 2).
    op : callable, optional
        Associative binary operation to apply (default is addition).

    Returns
    -------
    np.ndarray
        Local block updated with the prefix results including contributions from
        all lower-rank processes.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()    
    q = len(local_data)  # Local block size    

    # Step 1: Local prefix sum
    u = np.copy(local_data)
    for i in range(1, q):
        u[i] = op(u[i-1], u[i])    

    # Step 2: Prepare partial sums
    c = u[-1]  # Last element of local prefix
    d = u[0]   # First element of local prefix    
    num_steps = math.ceil(math.log(size, k+1))    

    for j in range(num_steps):
        # Determine destinations S_{i,j}
        destinations = []
        for offset in range(1, k+1):
            target = rank + offset * ((k+1) ** j)
            if target < size:
                destinations.append(target)        

        # Send c(i) to destinations
        for dest in destinations:
            comm.isend(c, dest=dest, tag=j)        

        # Determine sources R_{i,j}
        sources = []
        for offset in range(1, k+1):
            source = rank - offset * ((k+1) ** j)
            if source >= 0:
                sources.append(source)        

        # Receive and update
        temp = None
        for src in sources:
            recv_val = comm.recv(source=src, tag=j)
            temp = recv_val if temp is None else op(temp, recv_val)        

        if temp is not None:
            c = op(temp, c)
            d = op(temp, d)    

    # Step 3: Update local array using accumulated prefix
    if rank != 0:
        u = np.array([op(d, val) for val in u])

    return u
