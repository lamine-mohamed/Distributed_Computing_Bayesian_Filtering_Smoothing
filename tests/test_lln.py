import pytest
import numpy as np
from mpi4py import MPI
from scan_algorithms.lln import lln

# FILE: scan_algorithms/test_lln.py


@pytest.fixture
def mpi_env():
    """Fixture to provide MPI environment."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    return comm, rank, size


def test_lln_addition(mpi_env):
    comm, rank, size = mpi_env

    def add_op(x, y):
        return x + y

    local_value = np.array([rank + 1,rank + 2], dtype=int)
    result = lln(local_value, add_op)
    gathered_values = comm.gather(result, root=0)

    if rank == 0:
        expected_output = np.empty_like(gathered_values)
        expected_output[0] = np.array([1, 2], dtype=int)
        for r in range(1, size):
            expected_output[r] = add_op(expected_output[r - 1] , np.array([r + 1, r + 2], dtype=int))
        assert np.array_equal(gathered_values, expected_output), f"Expected {expected_output}, got {gathered_values}"


def test_lln_multiplication(mpi_env):
    comm, rank, size = mpi_env

    def mul_op(x, y):
        return x * y

    local_value = np.array([rank + 1,rank+2], dtype=int)
    result = lln(local_value, mul_op)
    gathered_values = comm.gather(result, root=0)

    if rank == 0:
        expected_output = np.empty_like(gathered_values)
        expected_output[0] = np.array([1, 2], dtype=int)
        for r in range(1, size):
            expected_output[r] = mul_op(expected_output[r - 1] , np.array([r + 1, r + 2], dtype=int))
        # expected_prefix_product = np.cumprod([[np.array([r + 1, r + 2], dtype=int) for r in range(size)]])
        assert np.array_equal(gathered_values, expected_output), f"Expected {expected_output}, got {gathered_values}"


def test_lln_edge_case_empty_array(mpi_env):
    comm, rank, size = mpi_env

    def add_op(x, y):
        return x + y

    local_value = np.array([], dtype=int)
    result = lln(local_value, add_op)
    gathered_values = comm.gather(result, root=0)

    if rank == 0:
        assert all(len(arr) == 0 for arr in gathered_values), "Expected empty arrays, got non-empty arrays."


def test_lln_edge_case_single_element(mpi_env):
    comm, rank, size = mpi_env

    def add_op(x, y):
        return x + y

    local_value = np.array([rank + 1], dtype=int)
    result = lln(local_value, add_op)
    gathered_values = comm.gather(result, root=0)

    if rank == 0:
        gathered_values = np.concatenate(gathered_values)
        expected_prefix_sum =np.cumsum([np.array(r +1, dtype=int) for r in range(size)])
        assert np.array_equal(gathered_values, expected_prefix_sum), f"Expected {expected_prefix_sum}, got {gathered_values}"


def test_lln_matrix_addition(mpi_env):
    comm, rank, size = mpi_env

    def add_op(x, y):
        return x + y

    local_value = np.array([[rank + 1, rank + 2], [rank + 3, rank + 4]], dtype=int)
    result = lln(local_value, add_op)
    gathered_values = comm.gather(result, root=0)

    if rank == 0:

        expected_results = np.zeros((size, 2, 2), dtype=int)
        expected_results[0] = np.array([[1, 2], [3, 4]])
        for r in range(1, size):
            expected_results[r] = expected_results[r - 1] + np.array([[r + 1, r + 2], [r + 3, r + 4]])
        assert np.array_equal(gathered_values, expected_results), f"Expected {expected_results}, got {gathered_values}"


def test_lln_matrix_multiplication(mpi_env):
    comm, rank, size = mpi_env

    def matmul_op(x, y):
        return x @ y

    local_value = np.array([[rank + 1, rank + 2], [rank + 3, rank + 4]], dtype=int)
    result = lln(local_value, matmul_op)
    gathered_values = comm.gather(result, root=0)

    if rank == 0:
        expected_results = np.zeros((size, 2, 2), dtype=int)
        expected_results[0] = np.array([[1, 2], [3, 4]])
        for r in range(1, size):
            expected_results[r] = expected_results[r - 1] @ np.array([[r + 1, r + 2], [r + 3, r + 4]])
        assert np.array_equal(gathered_values, expected_results), f"Expected {expected_results}, got {gathered_values}"