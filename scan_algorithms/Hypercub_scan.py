from mpi4py import MPI
import math


def hypercube_scan(my_element, OP):
    """
    Perform an **inclusive parallel prefix scan** using a hypercube communication pattern.

    This algorithm arranges MPI ranks as vertices of a d-dimensional hypercube 
    (d = ⌈log₂(p)⌉, where p is the number of ranks).  
    At each step, ranks exchange partial results with a partner determined 
    by flipping one bit of the rank ID.  
    The method supports arbitrary (non-power-of-two) communicator sizes by 
    skipping partners outside the range.

    Parameters
    ----------
    my_element : object
        The local value at this MPI process.
        Must be compatible with the binary operation `OP`.
    OP : callable
        Binary associative operation to apply.
        Must support the signature `OP(a, b)` returning the combined result.

    Returns
    -------
    object
        Inclusive prefix result for this rank .

    """
    comm = MPI.COMM_WORLD
    my_id = comm.Get_rank()
    p = comm.Get_size()

    # Number of hypercube dimensions (ceil allows non-complete cube)
    d = int(math.ceil(math.log2(p)))

    result = my_element
    msg = my_element

    for i in range(d):
        partner = my_id ^ (1 << i)  # Bitwise XOR to find partner

        if partner >= p:
            continue  # Skip invalid partners
        
        if i == d - 1:
            # Last round: one-way send/recv to avoid overwrite
            if my_id < partner:
                comm.send(msg, dest=partner, tag=i)
            else:
                element = comm.recv(source=partner, tag=i)
        else:
            # Two-way exchange
            element = comm.sendrecv(msg, dest=partner, sendtag=i,
                                    source=partner, recvtag=i)
            msg = OP(msg, element)

        if partner < my_id:  # Inclusive scan rule
            result = OP(result, element)

    return result


def Hypercube_Scan_exclusive(my_element, OP, **kwargs):
    """
    Perform an **exclusive parallel prefix scan** using a hypercube communication pattern.

    This algorithm arranges MPI ranks as vertices of a d-dimensional hypercube 
    (d = ⌈log₂(p)⌉, where p is the number of ranks).  
    At each step, ranks exchange partial results with a partner determined 
    by flipping one bit of the rank ID.  
    The method supports arbitrary (non-power-of-two) communicator sizes by 
    skipping partners outside the range.

    Parameters
    ----------
    my_element : object
        The local value at this MPI process.
        Must be compatible with the binary operation `OP`.
    OP : callable
        Binary associative operation to apply.
        Must support the signature `OP(a, b)` returning the combined result.
        Must also define an `identity` attribute representing the neutral element.
    comm : MPI.Comm, optional
        Custom MPI communicator (defaults to `MPI.COMM_WORLD`).

    Returns
    -------
    object
        Exclusive prefix result for this rank — i.e., the combination of 
        all elements from rank 0 up to (but not including) this rank.
    """
    comm = kwargs.get('comm', MPI.COMM_WORLD)
    my_id = comm.Get_rank()
    p = comm.Get_size()

    # Number of hypercube dimensions (ceil allows non-complete cube)
    d = int(math.ceil(math.log2(p)))

    result = OP.identity
    msg = my_element

    for i in range(d):
        partner = my_id ^ (1 << i)
        if partner >= p:
            continue

        tag = i
        req_send = None
        recv_buf = None

        if i == d - 1:
            if my_id < partner:
                req_send = comm.isend(msg, dest=partner, tag=tag)
            else:
                recv_buf = comm.recv(source=partner, tag=tag)
        else:
            req_send = comm.isend(msg, dest=partner, tag=tag)
            recv_buf = comm.recv(source=partner, tag=tag)
            msg = OP(msg, recv_buf)

        if partner < my_id:  # Exclusive scan rule
            result = OP(result, recv_buf if recv_buf is not None else my_element)

        if req_send:
            req_send.wait()

    return result
