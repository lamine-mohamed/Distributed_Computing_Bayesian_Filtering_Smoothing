"""
MPI Benchmarking Script for Filtering & Smoothing Operators.

This script measures:
1. The computation time for the `filtering_operator` and `smoothing_operator`
   from ParallelKF_tf on CPU and GPU for varying vector sizes.
2. The MPI communication time to transfer the required data structures
   between two MPI ranks.

It then plots the time evolution of computation and communication.
"""

import os
# Use async GPU memory allocator
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
import sys
import time
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from mpi4py import MPI
from scripts.ParallelKF_tf import filtering_operator, smoothing_operator



def run_benchmark_operations(max_d=300, filtering=True, smoothing=True):
    """
    Benchmark the computation time of filtering and smoothing operators
    for different state dimensions.

    Parameters
    ----------
    max_d : int
        Maximum state dimension to test.
    filtering : bool
        If True, measure filtering_operator performance.
    smoothing : bool
        If True, measure smoothing_operator performance.

    Returns
    -------
    sizes : np.ndarray
        Tested state dimensions.
    filtering_times : list of float
        Execution times for filtering_operator.
    smoothing_times : list of float
        Execution times for smoothing_operator.
    """
    sizes = np.logspace(2, np.log10(max_d), num=20, dtype=int)
    filtering_times = []
    smoothing_times = []

    for d in sizes:
        print(f"Testing size {d}")

        if filtering:
            # Random tensors for filtering
            A1 = tf.random.normal((d, d))
            b1 = tf.random.normal((d,))
            C1 = tf.random.normal((d, d))
            J1 = tf.random.normal((d, d))
            eta1 = tf.random.normal((d,))

            A2 = tf.random.normal((d, d))
            b2 = tf.random.normal((d,))
            C2 = tf.random.normal((d, d))
            J2 = tf.random.normal((d, d))
            eta2 = tf.random.normal((d,))

            # Warm-up
            filtering_operator(((A1, b1, C1, J1, eta1),
                                (A2, b2, C2, J2, eta2)))

            # Time measurement
            start = time.perf_counter()
            filtering_operator(((A1, b1, C1, J1, eta1),
                                (A2, b2, C2, J2, eta2)))
            filtering_times.append(time.perf_counter() - start)

        if smoothing:
            # Random tensors for smoothing
            E1 = tf.random.normal((d, d))
            g1 = tf.random.normal((d,))
            L1 = tf.random.normal((d, d))

            E2 = tf.random.normal((d, d))
            g2 = tf.random.normal((d,))
            L2 = tf.random.normal((d, d))

            # Warm-up
            smoothing_operator(((E1, g1, L1), (E2, g2, L2)))

            # Time measurement
            start = time.perf_counter()
            smoothing_operator(((E1, g1, L1), (E2, g2, L2)))
            smoothing_times.append(time.perf_counter() - start)

    return sizes, filtering_times, smoothing_times


def benchmark_communication(max_d=300, filtering=True, smoothing=True):
    """
    Benchmark MPI communication time for transferring filtering and smoothing
    data structures between two ranks.

    Parameters
    ----------
    max_d : int
        Maximum state dimension to test.
    filtering : bool
        If True, measure filtering data transfer.
    smoothing : bool
        If True, measure smoothing data transfer.

    Returns
    -------
    sizes : np.ndarray
        Tested state dimensions.
    filtering_transfer_times : list of float
        MPI transfer times for filtering data.
    smoothing_transfer_times : list of float
        MPI transfer times for smoothing data.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    sizes = np.logspace(2, np.log10(max_d), num=20, dtype=int)
    filtering_transfer_times = []
    smoothing_transfer_times = []

    for d in sizes:
        if filtering:
            if rank == 0:
                # Prepare filtering data
                A1 = tf.random.normal((d, d))
                b1 = tf.random.normal((d,))
                C1 = tf.random.normal((d, d))
                J1 = tf.random.normal((d, d))
                eta1 = tf.random.normal((d,))

                # Initial send
                comm.send((A1, b1, C1, J1, eta1), dest=1, tag=0)
                times = []
                for i in range(5):
                    comm.Barrier()
                    start = MPI.Wtime()
                    comm.send((A1, b1, C1, J1, eta1), dest=1, tag=i)
                    comm.Barrier()
                    times.append(MPI.Wtime() - start)
                filtering_transfer_times.append(np.mean(times))

            elif rank == 1:
                comm.recv(source=0, tag=0)
                for i in range(5):
                    comm.Barrier()
                    comm.recv(source=0, tag=i)
                    comm.Barrier()

        if smoothing:
            if rank == 0:
                # Prepare smoothing data
                E1 = tf.random.normal((d, d))
                g1 = tf.random.normal((d,))
                L1 = tf.random.normal((d, d))

                comm.send((E1, g1, L1), dest=1, tag=0)
                times = []
                for i in range(5):
                    comm.Barrier()
                    start = MPI.Wtime()
                    comm.send((E1, g1, L1), dest=1, tag=i)
                    comm.Barrier()
                    times.append(MPI.Wtime() - start)
                smoothing_transfer_times.append(np.mean(times))

            elif rank == 1:
                comm.recv(source=0, tag=0)
                for i in range(5):
                    comm.Barrier()
                    comm.recv(source=0, tag=i)
                    comm.Barrier()

    return sizes, filtering_transfer_times, smoothing_transfer_times


if __name__ == "__main__":
    max_size_log = int(sys.argv[1]) if len(sys.argv) > 1 else 14

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        with tf.device('/GPU:0'):
            print("Running on GPU")
            sizes, filtering_times, smoothing_times = run_benchmark_operations(max_d=9000, filtering=True, smoothing=True)
        with tf.device('/CPU'):
            print("Running on CPU")
            _, filtering_times_cpu, smoothing_times_cpu = run_benchmark_operations(max_d=9000, filtering=True, smoothing=True)


    comm.Barrier()
    tf.keras.backend.clear_session()

    _, filtering_transfer_times, smoothing_transfer_times = benchmark_communication(
        max_d=9000, filtering=True, smoothing=True
    )

    if rank == 0:
        # Filtering plot
        plt.figure(figsize=(10, 6))
        plt.plot(sizes, filtering_times, label="Filtering GPU Times", marker='o')
        plt.plot(sizes, filtering_times_cpu, label="Filtering CPU Times", marker='^')
        plt.plot(sizes, filtering_transfer_times, label="Filtering Communication", marker='x')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("State dimension")
        plt.ylabel("Time (seconds)")
        plt.title("Filtering: Computation vs Communication Times")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.show()

        # Smoothing plot
        plt.figure(figsize=(10, 6))
        plt.plot(sizes, smoothing_times, label="Smoothing GPU Times", marker='o')
        plt.plot(sizes, smoothing_times_cpu, label="Smoothing CPU Times", marker='^')
        plt.plot(sizes, smoothing_transfer_times, label="Smoothing Communication", marker='x')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("State dimension")
        plt.ylabel("Time (seconds)")
        plt.title("Smoothing: Computation vs Communication Times")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.show()
