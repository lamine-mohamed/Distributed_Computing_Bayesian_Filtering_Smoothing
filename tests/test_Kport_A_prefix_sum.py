import pytest
import numpy as np
from mpi4py import MPI
from scan_algorithms.Kport_A_prefix_sum import Kport_A_prefix_sum


@pytest.mark.mpi(min_size=2)

def test_kport_a_prefix_sum():
    """Test the Kport_A_prefix_sum algorithm with MPI."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Generate random data for each process
    

    # Run the Kport_A_prefix_sum algorithm
    result = Kport_A_prefix_sum(rank, op=lambda x, y: x + y)

    # Gather results from all processes
    gathered_results = comm.gather(result, root=0)

    if rank == 0:
        # Compute the reference prefix sum
        data = range(size )
        # result = np.concat(gathered_results)
        expected_results = np.cumsum(data).tolist()
        assert gathered_results == expected_results, f"Expected {expected_results}, got {result}"