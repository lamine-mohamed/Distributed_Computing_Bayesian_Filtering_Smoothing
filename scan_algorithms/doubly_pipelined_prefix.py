from mpi4py import MPI
import numpy as np
import sys
import math
from scripts.Binary_tree import construct_inorder_numbered_tree, get_node_with_index

def doubly_pipelined_binary_tree_scan(x_local, op, **kwargs):
    """
    Perform a **doubly pipelined** parallel prefix scan over distributed data 
    using a binary tree communication pattern.

    This algorithm splits the local data into blocks and overlaps upward and 
    downward communication phases to maximize parallelism.

    Parameters
    ----------
    x_local : np.ndarray
        The local input array on this MPI process.
    op : callable
        Binary associative operation to apply.
        Must support elementwise operations on numpy arrays of `x_local` shape.
    b : int, optional
        Number of blocks for pipelining. Defaults to `sqrt(m * n)` where:
            - m = local array length
            - n = tree height (≈ log₂(size))
        Will be adjusted downward until `m % b == 0`.
    comm : MPI.Comm, optional
        Custom MPI communicator (defaults to `MPI.COMM_WORLD`).

    Returns
    -------
    np.ndarray
        The local scan result after processing all ranks in prefix order.

    Notes
    -----
    The algorithm runs in **three phases**:

    1. **Fill Phase**: 
       - Data flows upward from leaves to root.
       - Each block is combined with results from children before sending up.

    2. **Steady-State Phase**:
       - Overlaps upward and downward communication.
       - Nodes receive partial sums from parents and propagate to children 
         while continuing upward sends.

    3. **Drain Phase**:
       - Completes the downward pass to all leaves after upward sends finish.

    The binary tree topology is constructed from rank IDs using an 
    in-order numbering scheme for balanced communication.
    """
    comm = kwargs.get('comm', MPI.COMM_WORLD)
    rank = comm.Get_rank()
    size = comm.Get_size()

    m = len(x_local)
    if m == 0:
        return x_local

    # Determine tree height in communication rounds
    n = int(np.ceil(np.log2(size + 1))) - 1 if size > 1 else 0

    # Determine block count (b)
    b = kwargs.get('b', int(np.sqrt(m * n)) if size > 1 else 1)
    if b > m:
        b = m
    while m % b != 0:
        b -= 1

    # Build binary tree topology
    tree = construct_inorder_numbered_tree(size)
    node = get_node_with_index(tree, rank)

    parent = node.parent.index if node.parent else None
    left = node.left.index if node.left else None
    right = node.right.index if node.right else None
    d = min(int(node.depth * 2 - 1), b) if node.depth > 0 else 0
    is_on_leftmost_path = node.is_on_leftmost_path
    is_on_rightmost_path = node.is_on_rightmost_path

    # Split into pipeline blocks
    x_blocks = np.split(x_local, b)
    L_blocks = x_blocks.copy()

    RLC = 0  # Upward communication block index
    RPC = 0  # Downward communication block index

    tag_base = lambda sender, blk: int(sender * 10**math.ceil(math.log10(b)) + blk)

    # -----------------------------
    # Phase 1: Fill Phase (Upward only)
    # -----------------------------
    send_requests = []
    for _ in range(d):
        if left is not None:
            L_blocks[RLC] = np.empty_like(x_blocks[RLC])
            comm.Recv(L_blocks[RLC], source=left, tag=tag_base(left, RLC))
            x_blocks[RLC] = op(L_blocks[RLC], x_blocks[RLC])
            L_blocks[RLC] = x_blocks[RLC]

        if right is not None and not is_on_rightmost_path:
            R_block = np.empty_like(x_blocks[RLC])
            comm.Recv(R_block, source=right, tag=tag_base(right, RLC))
            x_blocks[RLC] = op(x_blocks[RLC], R_block)

        if parent is not None and not is_on_rightmost_path:
            send_requests.append(comm.Isend(x_blocks[RLC], dest=parent, tag=tag_base(node.index, RLC)))

        RLC += 1
    MPI.Request.Waitall(send_requests)

    # -----------------------------
    # Phase 2: Steady-State Phase (Up + Down)
    # -----------------------------
    send_requests = []
    for _ in range(b - d):
        recv_requests = []

        if left is not None:
            L_blocks[RLC] = np.empty_like(x_blocks[RLC])
            recv_requests.append(comm.Irecv(L_blocks[RLC], source=left, tag=tag_base(left, RLC)))

        if right is not None and not is_on_rightmost_path:
            R_block = np.empty_like(x_blocks[RLC])
            recv_requests.append(comm.Irecv(R_block, source=right, tag=tag_base(right, RLC)))

        if parent is not None and not is_on_leftmost_path:
            P_block = np.empty_like(x_blocks[RPC])
            recv_requests.append(comm.Irecv(P_block, source=parent, tag=tag_base(parent, RPC)))

        # Wait & combine
        if left is not None:
            recv_requests[0].Wait()
            x_blocks[RLC] = op(L_blocks[RLC], x_blocks[RLC])
            L_blocks[RLC] = x_blocks[RLC]

        if right is not None and not is_on_rightmost_path:
            recv_requests[1].Wait()
            x_blocks[RLC] = op(x_blocks[RLC], R_block)

        if parent is not None and not is_on_rightmost_path:
            send_requests.append(comm.Isend(x_blocks[RLC], dest=parent, tag=tag_base(node.index, RLC)))

        RLC += 1

        if parent is not None and not is_on_leftmost_path:
            recv_requests[-1].Wait()
            x_blocks[RPC] = op(P_block, L_blocks[RPC])
            if left is not None:
                send_requests.append(comm.Isend(P_block, dest=left, tag=tag_base(node.index, RPC)))
        else:
            x_blocks[RPC] = L_blocks[RPC]

        if right is not None:
            send_requests.append(comm.Isend(x_blocks[RPC], dest=right, tag=tag_base(node.index, RPC)))

        RPC += 1
    MPI.Request.Waitall(send_requests)

    # -----------------------------
    # Phase 3: Drain Phase (Down only)
    # -----------------------------
    send_requests = []
    if parent is not None and not is_on_leftmost_path and d > 0:
        P_block = np.empty_like(x_blocks[0])
        comm.Recv(P_block, source=parent, tag=tag_base(parent, RPC))
    else:
        P_block = None

    for _ in range(d - 1):
        if parent is not None and not is_on_leftmost_path:
            P_block_next = np.empty_like(x_blocks[RPC+1])
            req_P = comm.Irecv(P_block_next, source=parent, tag=tag_base(parent, RPC+1))

            if left is not None:
                send_requests.append(comm.Isend(P_block, dest=left, tag=tag_base(node.index, RPC)))

            x_blocks[RPC] = op(P_block, L_blocks[RPC])
            req_P.Wait()
            P_block = P_block_next
        else:
            x_blocks[RPC] = L_blocks[RPC]

        if right is not None:
            send_requests.append(comm.Isend(x_blocks[RPC], dest=right, tag=tag_base(node.index, RPC)))
        RPC += 1

    if d > 0:
        if parent is not None and not is_on_leftmost_path:
            if left is not None:
                send_requests.append(comm.Isend(P_block, dest=left, tag=tag_base(node.index, RPC)))
            x_blocks[RPC] = op(P_block, L_blocks[RPC])
        else:
            x_blocks[RPC] = L_blocks[RPC]

        if right is not None:
            send_requests.append(comm.Isend(x_blocks[RPC], dest=right, tag=tag_base(node.index, RPC)))
        RPC += 1

    MPI.Request.Waitall(send_requests)
    comm.Barrier()

    return np.concatenate(x_blocks)
