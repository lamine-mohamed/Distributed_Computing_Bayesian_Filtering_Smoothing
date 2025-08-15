# pipelined_binary_tree_scan.py

from mpi4py import MPI
import numpy as np
import sys
from scripts.Binary_tree import construct_inorder_numbered_tree, get_node_with_index


def pipelined_binary_tree_scan(x_local, op):
    """
    Perform a pipelined binary tree scan (prefix operation) using MPI.

    This function implements a two-phase (up-phase and down-phase) scan
    over a binary tree of MPI processes. The operation is applied in a 
    hierarchical manner, allowing efficient parallel prefix computations.

    Parameters
    ----------
    x_local : scalar, list, or tensor-like
        The local value at this process to be included in the scan.
    op : callable
        A binary operation function that supports the data type of x_local.
        Must be associative (e.g., sum, max).

    Returns
    -------
    x_local : scalar, list, or tensor-like
        The updated local value after applying the prefix scan.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    tree = construct_inorder_numbered_tree(size)
    node = get_node_with_index(tree, rank)

    parent = node.parent.index if node.parent else None
    left = node.left.index if node.left else None
    right = node.right.index if node.right else None

    is_on_leftmost_path = node.is_on_leftmost_path
    is_on_rightmost_path = node.is_on_rightmost_path

    # Up-phase: propagate values from children to parent
    L_j = x_local
    if left is not None:
        L_j = comm.recv(source=left, tag=0)
        x_local = op(L_j, x_local)
        L_j = x_local # Update local value with left child's result

    if right is not None and not is_on_rightmost_path:
        R_j = comm.recv(source=right, tag=0)
        x_local = op(x_local, R_j)

    if parent is not None and not is_on_rightmost_path:
        comm.send(x_local, dest=parent, tag=0)

    # Down-phase: propagate values from parent to children
    if parent is not None and not is_on_leftmost_path:
        P_j = comm.recv(source=parent, tag=1)
        x_local = op(P_j, L_j)
        if left is not None:
            comm.send(P_j, dest=left, tag=1)
    else:
        x_local = L_j

    if right is not None:
        comm.send(x_local, dest=right, tag=1)

    return x_local


def exclusive_pipelined_binary_tree_scan(input_value, op, comm=None):
    """
    Perform an exclusive pipelined binary tree scan over MPI processes.

    The exclusive scan returns, for each process, the prefix combination 
    of all preceding elements according to the binary operation `op`. 
    The scan is implemented with an up-phase (leaf-to-root) and down-phase
    (root-to-leaf) communication over the process binary tree.

    Parameters
    ----------
    input_value : scalar, list, or tensor-like
        Local value at this process to include in the scan.
    op : callable
        Binary operation function supporting input_value.
        Must be associative and have an identity element: `op.identity`.
    comm : MPI.Comm, optional
        MPI communicator to use. Defaults to MPI.COMM_WORLD.

    Returns
    -------
    result : scalar, list, or tensor-like
        The exclusive scan result for this process.
    """
    if comm is None:
        comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    tree = construct_inorder_numbered_tree(size)
    node = get_node_with_index(tree, rank)

    parent = node.parent.index if node.parent else None
    left = node.left.index if node.left else None
    right = node.right.index if node.right else None

    is_on_leftmost_path = node.is_on_leftmost_path
    is_on_rightmost_path = node.is_on_rightmost_path

    x_local = input_value
    L_j = None
    send_req = None

    # Up-phase: receive from left and right children
    if left is not None:
        L_j = comm.recv(source=left, tag=0)
        x_local = op(L_j, x_local)

    if right is not None and not is_on_rightmost_path:
        R_j = comm.recv(source=right, tag=0)
        x_local = op(x_local, R_j)

    if parent is not None and not is_on_rightmost_path:
        comm.send(x_local, dest=parent, tag=0)

    # Down-phase: receive from parent and propagate to children
    if parent is not None and not is_on_leftmost_path:
        P_j = comm.recv(source=parent, tag=1)
        x_local = op(P_j, L_j) if L_j is not None else P_j
        if left is not None:
            send_req = comm.isend(P_j, dest=left, tag=1)
    else:
        x_local = L_j if L_j is not None else op.identity

    if right is not None:
        if L_j is not None:
            L_j = op(x_local, input_value)
        if send_req is not None:
            send_req.wait()
        comm.send(L_j, dest=right, tag=1)

    return x_local


from mpi4py import MPI
import numpy as np


def pipelined_binary_tree_scan_blocked(x_local, op, **kwargs):
    """
    Perform a pipelined parallel scan (prefix operation) using a binary tree communication pattern.

    The local data is split into `b` blocks, which are processed in a pipeline fashion
    over a binary tree topology. Each process exchanges partial results with its tree
    neighbors in two phases:
    
    1. **Up-phase**: Partial results are propagated from leaves to the root.
    2. **Down-phase**: Accumulated results are propagated from the root back to the leaves.

    Parameters
    ----------
    x_local : ndarray
        The local input array on this MPI process.
    op : callable
        A binary operation that combines two arrays elementwise.
        Must be associative.
    **kwargs : dict, optional
        Additional keyword arguments:
        
        - b : int, optional  
          Number of blocks to split `x_local` into for pipelining.  
          If not provided, it is chosen automatically based on input size and process count.
        - comm : MPI.Comm, optional  
          MPI communicator. Defaults to `MPI.COMM_WORLD`.

    Returns
    -------
    x_result : ndarray
        The result of the exclusive scan operation for this process's data segment.

    Notes
    -----
    - The binary tree is constructed using an inorder numbering scheme.
    - The operation `op` must be associative to ensure correctness.
    - This is a **blocked pipelined** version — splitting into `b` blocks enables overlap
      of communication and computation.
    - If `m = len(x_local)` is not divisible by `b`, `b` will be reduced until it divides `m`.
    """

    comm = kwargs.get('comm', MPI.COMM_WORLD)
    rank = comm.Get_rank()
    size = comm.Get_size()

    tree = construct_inorder_numbered_tree(size)
    node = get_node_with_index(tree, rank)

    # Tree relationships
    parent = node.parent.index if node.parent else None
    left = node.left.index if node.left else None
    right = node.right.index if node.right else None

    # Path position
    is_on_leftmost_path = node.is_on_leftmost_path
    is_on_rightmost_path = node.is_on_rightmost_path

    m = len(x_local)
    if m == 0:
        return x_local

    # Determine number of blocks
    if kwargs.get('b') is not None:
        b = kwargs['b']
    else:
        n = int(np.ceil(np.log2(size + 1))) - 1 if size > 1 else 0
        b = int(np.sqrt(m * n)) if size > 1 else 1
    while m % b != 0:
        b -= 1

    # Split into blocks
    x_blocks = np.split(x_local, b)
    L_blocks = x_blocks.copy()

    # -------- Up-phase --------
    for r in range(b):
        if left is not None:
            L_blocks[r] = comm.recv(source=left, tag=10 * b + r)
            x_blocks[r] = op(L_blocks[r], x_blocks[r])
            L_blocks[r] = x_blocks[r]

        if right is not None and not is_on_rightmost_path:
            R_block = comm.recv(source=right, tag=10 * b + r)
            x_blocks[r] = op(x_blocks[r], R_block)

        if parent is not None and not is_on_rightmost_path:
            comm.send(x_blocks[r], dest=parent, tag=10 * b + r)

    # -------- Down-phase --------
    for r in range(b):
        if parent is not None and not is_on_leftmost_path:
            P_block = comm.recv(source=parent, tag=10 * b + r)
            x_blocks[r] = op(P_block, L_blocks[r])
            if left is not None:
                comm.send(P_block, dest=left, tag=10 * b + r)
        else:
            x_blocks[r] = L_blocks[r]

        if right is not None:
            comm.send(x_blocks[r], dest=right, tag=10 * b + r)

    # Merge results
    return np.concatenate(x_blocks)
