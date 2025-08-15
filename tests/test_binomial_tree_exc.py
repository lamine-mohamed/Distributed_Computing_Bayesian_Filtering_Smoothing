import pytest
import numpy as np
from mpi4py import MPI
from scan_algorithms.binomial_tree import binomial_tree_exclusive_scan

# Import your scan function here or define it in the same file
# from your_module import binomial_tree_exclusive_scan

# Simple additive operator with identity
class AddOperator:
    def __call__(self, a, b):
        return a + b

    def identity(self):
        return 0

# Global MPI variables needed by your scan function
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# @pytest.mark.mpi(min_size=2)
def test_binomial_tree_exclusive_scan():
    OP = AddOperator()

    # Each process local input is its rank + 1
    local_val = rank + 1

    # Perform the exclusive scan using your implementation
    result = binomial_tree_exclusive_scan(local_val, OP)

    # Calculate expected exclusive prefix sum using MPI's built-in scan for reference
    expected_result = comm.exscan(local_val, op=MPI.SUM)

    # For rank 0, exscan returns None, so define expected as identity (0)
    if rank == 0:
        expected_result = OP.identity()

    # Compare result with expected (should be equal)
    assert result == expected_result, f"Rank {rank}: result={result} expected={expected_result}"
