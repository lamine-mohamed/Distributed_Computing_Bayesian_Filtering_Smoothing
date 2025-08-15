import pytest
import numpy as np
from mpi4py import MPI
from scan_algorithms.Linear_Pipeline import linear_pipeline_scan


@pytest.fixture
def mpi_env():
    """Fixture to provide MPI environment."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    return comm, rank, size

def test_linear_pipeline_addition(mpi_env):
    comm, rank, size = mpi_env

    def add_op(x, y):
        return x + y

    # n = int(np.ceil(np.log2(size+1))) - 1 if size > 1 else 0 
    # b = int(np.sqrt(4*n)) if size > 1 else 1
    # # Ensure b is a multiple of the input size
    # while 4 % b != 0:
    #     b -= 1


    local_array = np.arange(rank, rank + 4, dtype=int)
    # b = 2  # Number of blocks
    result = linear_pipeline_scan(local_array, add_op)
    gathered_values = comm.gather(result, root=0)

    if rank == 0:
        expected_output = np.zeros((size, 4), dtype=int)
        expected_output[0] = np.array([0, 1, 2, 3], dtype=int)
        for r in range(1, size):
            expected_output[r] = add_op(expected_output[r - 1], np.array([r, r + 1, r + 2, r + 3], dtype=int))
        assert np.array_equal(gathered_values, expected_output), f"Expected {expected_output}, got {gathered_values}"


def test_linear_pipeline_multiplication(mpi_env):
    comm, rank, size = mpi_env

    def mul_op(x, y):
        return x * y

    # n = int(np.ceil(np.log2(size+1))) - 1 if size > 1 else 0 
    # b = int(np.sqrt(4*n)) if size > 1 else 1
    # # Ensure b is a multiple of the input size
    # while 4 % b != 0:
    #     b -= 1

    local_array = np.arange(rank + 1, rank + 5, dtype=int)
    # b = 2  # Number of blocks
    result = linear_pipeline_scan(local_array, mul_op)
    gathered_values = comm.gather(result, root=0)

    if rank == 0:
        expected_output = np.zeros((size, 4), dtype=int)
        expected_output[0] = np.array([1, 2, 3, 4], dtype=int)
        for r in range(1, size):
            expected_output[r] = mul_op(expected_output[r - 1], np.array([r + 1, r + 2, r + 3, r + 4], dtype=int))
        assert np.array_equal(gathered_values, expected_output), f"Expected {expected_output}, got {gathered_values}"

def test_linear_pipeline_edge_case_empty_array(mpi_env):
    comm, rank, size = mpi_env

    def add_op(x, y):
        return x + y

    local_array = np.array([], dtype=int)
    # b = 1  # Number of blocks
    result = linear_pipeline_scan(local_array, add_op)
    gathered_values = comm.gather(result, root=0)

    if rank == 0:
        assert all(len(arr) == 0 for arr in gathered_values), f"Expected empty arrays, got non-empty arrays. {gathered_values}"


def test_linear_pipeline_edge_case_single_element(mpi_env):
    comm, rank, size = mpi_env

    def add_op(x, y):
        return x + y

    local_array = np.array([rank + 1], dtype=int)
    # b = 1  # Number of blocks
    result = linear_pipeline_scan(local_array, add_op)
    gathered_values = comm.gather(result, root=0)

    if rank == 0:
        gathered_values = np.concatenate(gathered_values)
        expected_prefix_sum = np.cumsum([r + 1 for r in range(size)])
        assert np.array_equal(gathered_values, expected_prefix_sum), f"Expected {expected_prefix_sum}, got {gathered_values}"


def test_linear_pipeline_matrix_addition(mpi_env):
    comm, rank, size = mpi_env

    def add_op(x, y):
        return x + y

    local_array = np.array([[rank + 1, rank + 2], [rank + 3, rank + 4]], dtype=int)
    # b = 2  # Number of blocks
    result = linear_pipeline_scan(local_array, add_op)
    gathered_values = comm.gather(result, root=0)

    if rank == 0:
        expected_results = np.zeros((size, 2, 2), dtype=int)
        expected_results[0] = np.array([[1, 2], [3, 4]])
        for r in range(1, size):
            expected_results[r] = expected_results[r - 1] + np.array([[r + 1, r + 2], [r + 3, r + 4]])
        assert np.array_equal(gathered_values, expected_results), f"Expected {expected_results}, got {gathered_values}"

def test_linear_pipeline_matrix_multiplication(mpi_env):
    comm, rank, size = mpi_env

    def matmul_op(x, y):
        return x @ y

    local_data = [np.array([[rank + i, rank + j] for j in range(2)]).reshape(2,2) for i in range(2)]
    # b = 2  # Number of blocks
    result = linear_pipeline_scan(local_data, matmul_op)
    gathered_values = comm.gather(result, root=0)

    if rank == 0:
        # expected_results = np.zeros((size, 2, 2), dtype=int)
        expected_results = np.empty((size, 2, 2, 2), dtype=int)
        expected_results[0] = [np.array([[0, 0],[0, 1]]).reshape(2,2), np.array([[1, 0],[1, 1]]).reshape(2,2)]
        for r in range(1, size):
            expected_results[r] = expected_results[r - 1] @ [np.array([[r + i, r + j] for j in range(2)]).reshape(2,2) for i in range(2)]
        assert np.array_equal(gathered_values, expected_results), f"Expected {expected_results}, got {gathered_values}"

