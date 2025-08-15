import pytest
import numpy as np
from mpi4py import MPI
from scan_algorithms.Pipelined_Binary_Tree import pipelined_binary_tree_scan_blocked
from scripts.Binary_tree import construct_inorder_numbered_tree, get_node_with_index





@pytest.fixture
def mpi_env():
    """Fixture to provide MPI environment."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    return comm, rank, size


def test_pipelined_binary_tree_blocked_addition(mpi_env):
    comm, rank, size = mpi_env

    def add_op(x, y):
        return x + y

    # Construct binary tree
    tree = construct_inorder_numbered_tree(size)
    node = get_node_with_index(tree, rank)

    # Local array for each process
    x_local = np.arange(rank * 4, (rank + 1) * 4)

    # Perform pipelined binary tree scan with addition
    result = pipelined_binary_tree_scan_blocked(x_local, add_op, b=2)

    # Gather results from all processes
    gathered_results = comm.gather(result, root=0)

    if rank == 0:
        # Compute expected prefix sum
        expected_results = np.zeros((size,4), dtype=int)
        expected_results[0] = np.arange(0, 4)
        for i in range(1,size):
            expected_results[i] = add_op(expected_results[i-1], np.arange(i * 4, (i + 1) * 4))
        # global_input = np.concatenate([np.arange(r * 4, (r + 1) * 4) for r in range(size)])
        # expected_results = np.cumsum(global_input,axis=0)
        np.testing.assert_array_equal(gathered_results, expected_results)


def test_pipelined_binary_tree_blocked_multiplication(mpi_env):
    comm, rank, size = mpi_env

    def mul_op(x, y):
        return x * y

    # Construct binary tree
    tree = construct_inorder_numbered_tree(size)
    node = get_node_with_index(tree, rank)

    # Local array for each process
    x_local = np.arange(1 + rank * 4, 1 + (rank + 1) * 4)

    # Perform pipelined binary tree scan with multiplication
    result = pipelined_binary_tree_scan_blocked(x_local, mul_op, b=2)

    # Gather results from all processes
    gathered_results = comm.gather(result, root=0)

    if rank == 0:
        # Compute expected prefix product
        expected_results = np.zeros((size,4), dtype=int)
        expected_results[0] = np.arange(1, 5)
        for i in range(1,size):
            expected_results[i] = mul_op(expected_results[i-1], np.arange(1 + i * 4, 1 + (i + 1) * 4))  
        np.testing.assert_array_equal(gathered_results, expected_results)


def test_pipelined_binary_tree_blocked_edge_case_empty_array(mpi_env):
    comm, rank, size = mpi_env

    def add_op(x, y):
        return x + y

    # Construct binary tree
    tree = construct_inorder_numbered_tree(size)
    node = get_node_with_index(tree, rank)

    # Local array is empty
    x_local = np.array([], dtype=int)

    # Perform pipelined binary tree scan
    result = pipelined_binary_tree_scan_blocked(x_local, add_op, b=1)

    # Gather results from all processes
    gathered_results = comm.gather(result, root=0)

    if rank == 0:
        # Expected result is an empty array
        np.testing.assert_array_equal(np.concatenate(gathered_results), np.array([], dtype=int))


def test_pipelined_binary_tree_blocked_edge_case_single_element(mpi_env):
    comm, rank, size = mpi_env

    def add_op(x, y):
        return x + y

    # Construct binary tree
    tree = construct_inorder_numbered_tree(size)
    node = get_node_with_index(tree, rank)

    # Local array has a single element
    x_local = np.array([rank + 1], dtype=int)

    # Perform pipelined binary tree scan
    result = pipelined_binary_tree_scan_blocked(x_local, add_op, b=1)

    # Gather results from all processes
    gathered_results = comm.gather(result, root=0)

    if rank == 0:
        # Compute expected prefix sum
        global_input = np.arange(1, size + 1)
        expected_results = np.cumsum(global_input)
        np.testing.assert_array_equal(np.concatenate(gathered_results), expected_results)


def test_pipelined_binary_tree_blocked_matrix_addition(mpi_env):
    comm, rank, size = mpi_env

    def add_op(x, y):
        return x + y

    # Construct binary tree
    tree = construct_inorder_numbered_tree(size)
    node = get_node_with_index(tree, rank)

    # Local matrix for each process
    x_local = np.array([[rank + 1, rank + 2], [rank + 3, rank + 4]])

    # Perform pipelined binary tree scan with matrix addition
    result = pipelined_binary_tree_scan_blocked(x_local, add_op, b=2)

    # Gather results from all processes
    gathered_results = comm.gather(result, root=0)

    if rank == 0:
        expected_results = np.zeros((size, 2, 2), dtype=int)
        expected_results[0] = np.array([[1, 2], [3, 4]])
        for r in range(1,size):
            expected_results[r] = expected_results[r - 1] + np.array([[r + 1, r + 2], [r + 3, r + 4]])
        np.testing.assert_array_equal(gathered_results, expected_results)