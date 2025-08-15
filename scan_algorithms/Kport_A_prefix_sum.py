from mpi4py import MPI
import math

def Kport_A_prefix_sum(local_data, k=2, op=lambda x, y: x + y):
    """
    Perform a **k-port parallel prefix sum** across MPI processes.

    This algorithm generalizes the binary tree prefix sum by allowing each process
    to communicate with up to `k` neighbors in each step.  
    It computes the prefix operation in **logₖ₊₁(p)** steps, where `p` is the number 
    of MPI processes.

    Parameters
    ----------
    local_data : object
        The initial local value ν(i) of this MPI process.
    k : int, optional
        Fan-out: number of processes each process sends to at each step.
        Default is 2 (binary tree).
    op : callable, optional
        Associative binary operation to apply (default is addition).
        Must support the signature `op(a, b)` and be associative.

    Returns
    -------
    result : object
        The final prefix result at this process — i.e., the combination of all
        values from rank 0 up to and including this rank according to `op`.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()    
    c = local_data    

    # Number of steps to complete the prefix
    num_steps = math.ceil(math.log(size, k+1))   

    for j in range(num_steps):
        # Determine destinations S_{i,j}
        destinations = []
        for offset in range(1, k+1):
            target = rank + offset * ((k+1) ** j)
            if target < size:
                destinations.append(target)   

        # Send c(i) to destinations asynchronously
        for dest in destinations:
            comm.isend(c, dest=dest, tag=j) 

        # Determine sources R_{i,j}
        sources = []
        for offset in range(1, k+1):
            source = rank - offset * ((k+1) ** j)
            if source >= 0:
                sources.append(source)   

        # Receive values from sources and update local cumulative value
        for src in sources:
            recv_val = comm.recv(source=src, tag=j)
            c = op(c, recv_val)    

    return c

