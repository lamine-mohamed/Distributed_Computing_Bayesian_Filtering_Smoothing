import pytest
import numpy as np
from mpi4py import MPI
from scan_algorithms.Pipelined_Binary_Tree import pipelined_binary_tree_scan
from scripts.Binary_tree import construct_inorder_numbered_tree, get_node_with_index



def op(a, b):
    """
    Example binary operation: addition.
    Modify this function to implement other operations (e.g., max, min, etc.)
    """
    return a + b



@pytest.fixture
def mpi_env():
    """Fixture to provide MPI environment."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    return comm, rank, size


def test_pipelined_binary_tree_addition(mpi_env):
    comm, rank, size = mpi_env

    # Construct binary tree
    tree = construct_inorder_numbered_tree(size)
    node = get_node_with_index(tree, rank)

    # Local value for each process
    x_local = rank + 1

    # Perform pipelined binary tree scan with addition
    result = pipelined_binary_tree_scan(x_local, lambda a, b: a + b)

    # Gather results from all processes
    gathered_results = comm.gather(result, root=0)

    if rank == 0:
        # Compute expected prefix sum
        expected_results = np.cumsum(np.arange(1, size + 1))
        np.testing.assert_array_equal(gathered_results, expected_results)


def test_pipelined_binary_tree_multiplication(mpi_env):
    comm, rank, size = mpi_env

    # Construct binary tree
    tree = construct_inorder_numbered_tree(size)
    node = get_node_with_index(tree, rank)

    # Local value for each process
    x_local = rank + 1

    # Perform pipelined binary tree scan with multiplication
    result = pipelined_binary_tree_scan(x_local, lambda a, b: a * b)

    # Gather results from all processes
    gathered_results = comm.gather(result, root=0)

    if rank == 0:
        # Compute expected prefix product
        expected_results = np.cumprod(np.arange(1, size + 1))
        np.testing.assert_array_equal(gathered_results, expected_results)


def test_pipelined_binary_tree_edge_case_single_node(mpi_env):
    comm, rank, size = mpi_env

    if size != 1:
        pytest.skip("This test is only valid for a single process.")

    # Construct binary tree with a single node
    tree = construct_inorder_numbered_tree(size)
    node = get_node_with_index(tree, rank)

    # Local value for the single process
    x_local = rank + 1

    # Perform pipelined binary tree scan
    result = pipelined_binary_tree_scan(x_local, lambda a, b: a + b)

    # Gather results from all processes
    gathered_results = comm.gather(result, root=0)

    if rank == 0:
        # Expected result is the local value itself
        expected_results = [x_local]
        np.testing.assert_array_equal(gathered_results, expected_results)


def test_pipelined_binary_tree_edge_case_empty_tree(mpi_env):
    comm, rank, size = mpi_env

    if size != 0:
        pytest.skip("This test is only valid for an empty tree.")

    # No tree to construct, no nodes
    node = None
    x_local = 0

    # Perform pipelined binary tree scan
    result = pipelined_binary_tree_scan(x_local, lambda a, b: a + b)

    # Gather results from all processes
    gathered_results = comm.gather(result, root=0)

    if rank == 0:
        # Expected result is an empty list
        expected_results = []
        np.testing.assert_array_equal(gathered_results, expected_results)

def test_pipelined_binary_tree_matrix_multiplication(mpi_env):
    comm, rank, size = mpi_env

    # Construct binary tree
    tree = construct_inorder_numbered_tree(size)
    node = get_node_with_index(tree, rank)

    # Local matrix for each process
    x_local = np.array([[rank + 1, rank + 2], [rank + 3, rank + 4]])

    # Perform pipelined binary tree scan with matrix multiplication
    result = pipelined_binary_tree_scan(x_local, lambda a, b: a @ b)

    # Gather results from all processes
    gathered_results = comm.gather(result, root=0)

    if rank == 0:
        # Compute expected prefix product for matrices
        expected_results = np.zeros((size, 2, 2), dtype=int)
        expected_results[0] = np.array([[1, 2], [3, 4]])
        for r in range(1, size):
            expected_results[r] = expected_results[r - 1] @ np.array([[r + 1, r + 2], [r + 3, r + 4]])
        np.testing.assert_array_equal(gathered_results, expected_results)
    
def test_pipelined_binary_tree_matrix_addition(mpi_env):
    comm, rank, size = mpi_env

    # Construct binary tree
    tree = construct_inorder_numbered_tree(size)
    node = get_node_with_index(tree, rank)

    # Local matrix for each process
    x_local = np.array([[rank + 1, rank + 2], [rank + 3, rank + 4]])

    # Perform pipelined binary tree scan with matrix addition
    result = pipelined_binary_tree_scan(x_local, lambda a, b: a + b)

    # Gather results from all processes
    gathered_results = comm.gather(result, root=0)

    if rank == 0:
        expected_results = np.zeros((size, 2, 2), dtype=int)
        expected_results[0] = np.array([[1, 2], [3, 4]])
        for r in range(1, size):
            expected_results[r] = expected_results[r - 1] + np.array([[r + 1, r + 2], [r + 3, r + 4]])
        np.testing.assert_array_equal(gathered_results, expected_results)