import pytest
from mpi4py import MPI
from scan_algorithms.Hypercub_scan import hypercube_scan
import numpy as np

@pytest.mark.mpi(min_size=2)

def test_hypercube_scan():
    """
    Test the hypercube scan algorithm with MPI.
    This test requires at least 2 MPI processes.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Each process will have a unique value to scan
    local_value = rank + 1

    # Perform the hypercube scan
    result = hypercube_scan(local_value, lambda x, y: x + y)

    # Check if the result is correct
    gathered_results = comm.gather(result, root=0)
    if rank == 0:
        # Compute the reference prefix sum
        data = range(1, size + 1)
        # result = np.concat(gathered_results)
        expected_results = np.cumsum(data).tolist()
        assert gathered_results == expected_results , f"Expected {expected_results}, got {gathered_results}"
   