import numpy as np
from mpi4py import MPI
import pytest
from scan_algorithms.doubly_pipelined_prefix import doubly_pipelined_binary_tree_scan  
import sys


@pytest.fixture
def mpi_env():
    """Fixture to provide MPI environment."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    return comm, rank, size

def test_doubly_pipelined_binary_tree_scan_addition(mpi_env):
    comm, rank, size = mpi_env

    # Simulate a vector with 4*size elements and split among processes
    
    local_data = np.arange(rank , rank + 4)
    gathered_input = comm.gather(local_data, root=0)

    # Perform scan with addition
    result = doubly_pipelined_binary_tree_scan(local_data, lambda x,y: x+y)

    # Gather all result vectors to root for validation
    global_results = comm.gather(result, root=0)

    if rank == 0:
        
        expected_results = np.empty_like(global_results)
        expected_results[0] = np.arange(0, 4)
        for i in range(1,size):
            expected_results[i] = np.array(expected_results[i-1] + gathered_input[i]) 


        assert np.array_equal(global_results, expected_results), f"Expected {expected_results}, got {global_results}"

def test_doubly_pipelined_binary_tree_scan_multiplication(mpi_env):
    comm, rank, size = mpi_env

    # Simulate a vector with 4*size elements and split among processes
    local_data = np.arange(rank + 1, rank + 5, dtype=int)

    # Perform scan with multiplication
    result = doubly_pipelined_binary_tree_scan(local_data, lambda x,y: x*y )

    # Gather all result vectors to root for validation
    global_results = comm.gather(result, root=0)

    if rank == 0:
        expected_results = np.empty_like(global_results)
        expected_results[0] = np.array([1, 2, 3, 4])
        for i in range(1,size):
            expected_results[i] = np.array(expected_results[i-1] * np.arange(i+1, i+5))  
        
        assert np.array_equal(global_results, expected_results), f"Expected {expected_results}, got {global_results}"


def test_doubly_pipelined_binary_tree_scan_edge_case_empty_array(mpi_env):
    comm, rank, size = mpi_env

    # Simulate an empty vector
    local_data = np.array([])

    # Perform scan with addition
    result = doubly_pipelined_binary_tree_scan(local_data, lambda x,y: x+y )

    # Gather all result vectors to root for validation
    global_results = comm.gather(result, root=0)

    if rank == 0:
        expected_results = np.empty_like(global_results)
        expected_results.fill(0)
        assert np.array_equal(global_results, expected_results), f"Expected {expected_results}, got {global_results}"

def test_doubly_pipelined_binary_tree_scan_matrix_multiplication(mpi_env):
    comm, rank, size = mpi_env

    # Simulate a vector with 4*size elements and split among processes
    local_data = np.random.rand(2, 2, 2)
    glopal_data = comm.gather(local_data, root=0)

    # Perform scan with multiplication
    result = doubly_pipelined_binary_tree_scan(local_data, lambda x,y: x@y )

    # Gather all result vectors to root for validation
    global_results = comm.gather(result, root=0)

    if rank == 0:
        expected_results = np.empty_like(global_results)
        expected_results[0] = glopal_data[0]
        for r in range(1,size):
            expected_results[r] = np.array(expected_results[r - 1] @ glopal_data[r])  
        # global_results = np.concatenate(global_results, axis=0)
        # expected_results = np.concatenate(expected_results, axis=0)
        # assert np.array_equal(global_results, expected_results), f"Expected {expected_results}, got {global_results}"
        assert np.allclose(global_results, expected_results), f"Expected {expected_results}, got {global_results}"
        
def test_doubly_pipelined_binary_tree_scan_matrix_addition(mpi_env):
    comm, rank, size = mpi_env

    # Simulate a vector with 4*size elements and split among processes
    local_data = np.random.rand(2, 2, 2)
    glopal_data = comm.gather(local_data, root=0)

    # Perform scan with addition
    result = doubly_pipelined_binary_tree_scan(local_data, lambda x,y: x+y )

    # Gather all result vectors to root for validation
    global_results = comm.gather(result, root=0)

    if rank == 0:
        expected_results = np.empty_like(global_results)
        expected_results[0] = glopal_data[0]
        for r in range(1,size):
            expected_results[r] = np.array(expected_results[r - 1] + glopal_data[r])  
        # global_results = np.concatenate(global_results, axis=0)
        # expected_results = np.concatenate(expected_results, axis=0)
        # assert np.array_equal(global_results, expected_results), f"Expected {expected_results}, got {global_results}"
        assert np.allclose(global_results, expected_results), f"Expected {expected_results}, got {global_results}"

def test_doubly_pipelined_binary_tree_scan_edge_case_empty_matrix(mpi_env):
    comm, rank, size = mpi_env

    # Simulate an empty vector
    local_data = np.empty((0, 2, 2))

    # Perform scan with addition
    result = doubly_pipelined_binary_tree_scan(local_data, lambda x,y: x+y )

    # Gather all result vectors to root for validation
    global_results = comm.gather(result, root=0)

    if rank == 0:
        expected_results = np.empty_like(global_results)
        expected_results.fill(0)
        assert np.array_equal(global_results, expected_results), f"Expected {expected_results}, got {global_results}"

def test_doubly_pipelined_binary_tree_scan_edge_case_single_element(mpi_env):
    comm, rank, size = mpi_env

    # Simulate a single element vector
    local_data = np.array([rank + 1], dtype=int)
    global_data = comm.gather(local_data, root=0)

    # Perform scan with addition
    result = doubly_pipelined_binary_tree_scan(local_data, lambda x,y: x+y )

    # Gather all result vectors to root for validation
    global_results = comm.gather(result, root=0)

    if rank == 0:
        expected_results = np.empty_like(global_results)
        expected_results[0] = global_data[0]
        for i in range(1, size):
            expected_results[i] = np.array(expected_results[i - 1] + global_data[i])
        assert np.array_equal(global_results, expected_results), f"Expected {expected_results}, got {global_results}"

