import pytest
import numpy as np
from mpi4py import MPI
from scan_algorithms.simultaneous_binomial_tree import simultaneous_binomial_tree_scan

@pytest.fixture
def mpi_env():
    """Fixture to provide MPI environment."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    return comm, rank, size


def test_simultaneous_binomial_tree_addition(mpi_env):
    comm, rank, size = mpi_env

    def add_op(x, y):
        return x + y

    local_val = rank + 1
    result = simultaneous_binomial_tree_scan(local_val, add_op)

    # Gather all results into a single array
    global_results = comm.gather(result, root=0)

    if rank == 0:
        expected_results = np.cumsum([r + 1 for r in range(size)])
        np.testing.assert_array_equal(global_results, expected_results)


def test_simultaneous_binomial_tree_multiplication(mpi_env):
    comm, rank, size = mpi_env

    def mul_op(x, y):
        return x * y

    local_val = rank + 1
    result = simultaneous_binomial_tree_scan(local_val, mul_op)

    global_results = comm.gather(result, root=0)

    if rank == 0:
        expected_results = np.cumprod([r + 1 for r in range(size)])
        np.testing.assert_array_equal(global_results, expected_results)


def test_simultaneous_binomial_tree_edge_case_single_process(mpi_env):
    comm, rank, size = mpi_env

    if size == 1:
        def add_op(x, y):
            return x + y

        local_val = rank + 1
        result = simultaneous_binomial_tree_scan(local_val, add_op)

        assert result == local_val


def test_simultaneous_binomial_tree_edge_case_zero_value(mpi_env):
    comm, rank, size = mpi_env

    def add_op(x, y):
        return x + y

    local_val = 0
    result = simultaneous_binomial_tree_scan(local_val, add_op)

    global_results = comm.gather(result, root=0)

    if rank == 0:
        expected_results = [0] * size
        np.testing.assert_array_equal(global_results, expected_results)

def test_simultaneous_binomial_tree_matrix_multiplication(mpi_env):
    comm, rank, size = mpi_env

    def matmul_op(x, y):
        return x @ y

    local_val = np.array([[rank + 1, rank + 2], [rank + 3, rank + 4]])
    result = simultaneous_binomial_tree_scan(local_val, matmul_op)

    global_results = comm.gather(result, root=0)

    if rank == 0:
        # Construct the expected results for matrix multiplication
        expected_results = np.zeros((size, 2, 2), dtype=int)
        expected_results[0] = np.array([[1, 2], [3, 4]])    
        for r in range(1,size):
            expected_results[r] = expected_results[r - 1] @ np.array([[r + 1, r + 2], [r + 3, r + 4]])  
        
        np.testing.assert_array_equal(global_results, expected_results)

def test_simultaneous_binomial_tree_matrix_addition(mpi_env):
    comm, rank, size = mpi_env

    def add_op(x, y):
        return x + y

    local_val = np.array([[rank + 1, rank + 2], [rank + 3, rank + 4]])
    result = simultaneous_binomial_tree_scan(local_val, add_op)

    global_results = comm.gather(result, root=0)

    if rank == 0:
        expected_results = np.zeros((size, 2, 2), dtype=int)
        expected_results[0] = np.array([[1, 2], [3, 4]])    
        for r in range(1,size):
            expected_results[r] = expected_results[r - 1] + np.array([[r + 1, r + 2], [r + 3, r + 4]])  
        
        np.testing.assert_array_equal(global_results, expected_results)