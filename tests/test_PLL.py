import sys
import pytest
import numpy as np
from mpi4py import MPI
from scan_algorithms.pll import PLL


@pytest.fixture
def mpi_env():
    """Fixture to provide MPI environment."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    return comm, rank, size


def test_pll_addition(mpi_env):
    comm, rank, size = mpi_env

    def add_op(x, y):
        return x + y

    local_arr = np.array([rank + 1, rank + 2], dtype=int)

    PLL(local_arr, add_op)
    global_arr = comm.gather(local_arr, root=0)
    if rank == 0:
        global_arr = np.concatenate(global_arr)
        expected_input = np.concatenate([np.array([r + 1, r + 2], dtype=int) for r in range(size)])
        expected_prefix_sum = np.cumsum(expected_input)
        assert np.array_equal(global_arr, expected_prefix_sum), f"Expected {expected_prefix_sum}, got {global_arr}"


def test_pll_multiplication(mpi_env):
    comm, rank, size = mpi_env

    def mul_op(x, y):
        return x * y

    local_arr = np.arange(rank * 2 + 1, rank * 2 + 3, dtype=int)

    PLL(local_arr, mul_op)
    global_arr = comm.gather(local_arr, root=0)
    if rank == 0:
        global_arr = np.concatenate(global_arr)
        expected_output = np.cumprod(np.concatenate([np.arange(r * 2 + 1, r * 2 + 3, dtype=int) for r in range(size)]))
        assert np.array_equal(global_arr, expected_output), f"Expected {expected_output}, got {global_arr}"


def test_pll_edge_case_empty_array(mpi_env):
    comm, rank, size = mpi_env

    def add_op(x, y):
        return x + y

    local_arr = np.array([], dtype=int)

    PLL(local_arr, add_op)
    global_arr = comm.gather(local_arr, root=0)
    if rank == 0:
        global_arr = np.concatenate(global_arr)
        assert global_arr.size == 0, "Expected empty array, got non-empty array."


def test_pll_edge_case_single_element(mpi_env):
    comm, rank, size = mpi_env

    def add_op(x, y):
        return x + y

    local_arr = np.array([rank + 1], dtype=int)

    PLL(local_arr, add_op)
    global_arr = comm.gather(local_arr, root=0)

    if rank == 0:
        global_arr = np.concatenate(global_arr)
        expected_input = np.arange(1, size + 1)
        expected_prefix_sum = np.cumsum(expected_input)
        assert np.array_equal(global_arr, expected_prefix_sum), f"Expected {expected_prefix_sum}, got {global_arr}"


def test_pll_matrix_addition(mpi_env):
    comm, rank, size = mpi_env

    def add_op(x, y):
        return x + y
    # Construct an array of matrices for each rank
    local_arr = np.array([[[rank + 1, rank + 2], [rank + 3, rank + 4]]], dtype=int)
    # local_arr = np.array([[rank + 1, rank + 2], [rank + 3, rank + 4]])

    PLL(local_arr, add_op)

    global_arr = comm.gather(local_arr, root=0)

    if rank == 0:
        global_arr = np.concatenate(global_arr)
        expected_results = np.zeros((size, 2, 2), dtype=int)
        expected_results[0] = np.array([[1, 2], [3, 4]])
        for r in range(1, size):
            expected_results[r] = expected_results[r - 1] + np.array([[r + 1, r + 2], [r + 3, r + 4]])
        assert np.array_equal(global_arr, expected_results), f"Expected {expected_results}, got {global_arr}"


def test_pll_matrix_multiplication(mpi_env):
    comm, rank, size = mpi_env

    def matmul_op(x, y):
        return x @ y

    local_arr = np.array([[[rank + 1, rank + 2], [rank + 3, rank + 4]]], dtype=int)

    # local_arr = np.array([[rank + 1, rank + 2], [rank + 3, rank + 4]])

    PLL(local_arr, matmul_op)

    global_arr = comm.gather(local_arr, root=0)

    if rank == 0:
        global_arr = np.concatenate(global_arr)
        expected_results = np.zeros((size, 2, 2), dtype=int)
        expected_results[0] = np.array([[1, 2], [3, 4]])
        for r in range(1, size):
            expected_results[r] = expected_results[r - 1] @ np.array([[r + 1, r + 2], [r + 3, r + 4]])
        assert np.array_equal(global_arr, expected_results), f"Expected {expected_results}, got {global_arr}"


# if __name__ == "__main__":
#     pytest.main()
#     exit(0)