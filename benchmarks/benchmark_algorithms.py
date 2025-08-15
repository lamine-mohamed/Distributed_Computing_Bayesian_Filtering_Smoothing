import json
import os
import sys
from mpi4py import MPI
import numpy as np
import time
import matplotlib.pyplot as plt
from scan_algorithms.pll import PLL
from scan_algorithms.Linear_Pipeline import linear_pipeline_scan
from scan_algorithms.doubly_pipelined_prefix import doubly_pipelined_binary_tree_scan
from scan_algorithms.Pipelined_Binary_Tree import pipelined_binary_tree_scan, pipelined_binary_tree_scan_blocked
from scan_algorithms.binomial_tree import binomial_tree_scan
from scan_algorithms.simultaneous_binomial_tree import simultaneous_binomial_tree_scan
from scan_algorithms.lln import lln
from scan_algorithms.Hypercub_scan import hypercube_scan
from scan_algorithms.recursive_doubling_scan import recursive_doubling_scan
from scripts.utils import benchmark_algorithm


# def benchmark_case1(algorithms, op, **kwargs):
    

#     # Generate local data
#     inputs = {
#         "int": np.array([rank * np.log(2)]),
#         "double": np.array([rank * np.log(2)], dtype=np.float64),
#         "matrix": np.ones((input_size, input_size), dtype=np.float64),

#     }

#     results = []
#     for data_type, local_data in inputs.items():
#         # Benchmark the algorithm
#         for alg_name, algorithm in algorithms.items():
#             try:
#                 algo, kwargs = algorithm["algo"], algorithm.get("kwargs", {})
#                 time_taken, result = benchmark_algorithm(algo, local_data, op, **kwargs)
#                 if rank == 0:
#                     results.append({
#                         "mpi_size": size,
#                         "input_size": input_size,
#                         "data_type": data_type,
#                         "algorithm": alg_name,
#                         "time": time_taken,
#                     })
#                     # os.makedirs("results", exist_ok=True)
#                     # with open("results/case1_results.json", "w") as f:
#                     #     json.dump(results, f, indent=4)
                    
#                     print(f"Algorithm: {alg_name}, Time taken for {data_type}: {time_taken:.6f} seconds")
#                     sys.stdout.flush()
#             except Exception as e:
#                 if rank == 0:
#                     print(f"Error in {alg_name} for {data_type}: {e}, line: {e.__traceback__.tb_lineno}")
#                     sys.stdout.flush()
#                 exit(1)
#     # Write all results to a single JSON file
#     if rank == 0:
#         os.makedirs("results", exist_ok=True)
#         json_file = "results/case1_results.json"
#         if os.path.exists(json_file) and os.path.getsize(json_file) > 0:
#             # Load existing results and append new ones
#             with open(json_file, "r") as f:
#                 existing_results = json.load(f)
#         else:
#             existing_results = []

#         existing_results.extend(results)

#         with open(json_file, "w") as f:
#             json.dump(existing_results, f, indent=4)
#     return results

def benchmark_victor_input(algorithms, op, input_size, json_file=None):
    """
    Benchmark multiple algorithms with uniform vector inputs.

    This benchmark is intended for larger, fixed-size vector inputs
    that are identical across all MPI processes.

    Parameters
    ----------
    algorithms : dict
        Mapping from algorithm name to a dict containing:
        - "algo": callable scan function
        - "kwargs": optional dict of parameters for the algorithm.
    op : callable
        Binary operation to use for the scan.
    input_size : int
        Number of elements in the local vector for each process.
    json_file : str, optional
        Path to the JSON file where results will be stored.

    Returns
    -------
    results : list of dict
        Each dict contains:
        - "mpi_size": total number of processes
        - "input_size": number of elements per process
        - "data_type": string describing the type (e.g., "float")
        - "algorithm": algorithm name
        - "time": average execution time in seconds (mean of last 4 runs)

    Notes
    -----
    - The first run is discarded to avoid initialization overhead.
    - Results are appended to the provided `json_file`.
    """
    inputs = {"float": np.ones(input_size, dtype=np.float64)}

    results = []
    for data_type, local_data in inputs.items():
        times = []
        for alg_name, algorithm in algorithms.items():
            try:
                algo, algo_kwargs = algorithm["algo"], algorithm.get("kwargs", {})
                for i in range(5):
                    time_taken, _ = benchmark_algorithm(algo, local_data, op, **algo_kwargs)
                    if i > 0:
                        times.append(time_taken)
                if times:
                    avg_time = np.mean(times)
                if rank == 0:
                    results.append({
                        "mpi_size": size,
                        "input_size": input_size,
                        "data_type": data_type,
                        "algorithm": alg_name,
                        "time": avg_time,
                    })
                    print(f"Algorithm: {alg_name}, Time taken for {data_type}: {avg_time:.6f} seconds")
                    sys.stdout.flush()
            except Exception as e:
                if rank == 0:
                    print(f"Error in {alg_name} for {data_type}: {e}, line: {e.__traceback__.tb_lineno}")
                    sys.stdout.flush()
                exit(1)

    if rank == 0:
        existing_results = []
        if os.path.exists(json_file) and os.path.getsize(json_file) > 0:
            with open(json_file, "r") as f:
                existing_results = json.load(f)
        existing_results.extend(results)
        with open(json_file, "w") as f:
            json.dump(existing_results, f, indent=4)
    return results

def benchmark_b(op, input_size, b, json_file=None):
    """
    Benchmark algorithms with a specified block size.

    Designed to compare blocked and pipelined scan algorithms by
    fixing the number of blocks `b` into which the local input is split.

    Parameters
    ----------
    op : callable
        Binary operation to use for the scan.
    input_size : int
        Size of each dimension for the local matrix (input is `input_size x input_size`).
    b : int
        Number of blocks for pipelining.
    json_file : str, optional
        Path to the JSON file where results will be stored.

    Returns
    -------
    results : list of dict
        Each dict contains:
        - "block_size": b
        - "data_type": string describing the type (e.g., "matrix")
        - "algorithm": algorithm name
        - "time": average execution time in seconds (mean of last 4 runs)

    Notes
    -----
    - The first run is discarded to avoid warm-up effects.
    - Currently benchmarks:
        * Linear Pipeline
        * Doubly Pipelined Binary Tree
        * Pipelined Binary Tree Blocked
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    inputs = {"matrix": np.ones((input_size, input_size), dtype=np.float64)}

    algorithms = {
        "Linear Pipeline": {"algo": linear_pipeline_scan, "kwargs": {'b': b}},
        "Doubly Pipelined Binary Tree": {"algo": doubly_pipelined_binary_tree_scan, "kwargs": {'b': b}},
        "Pipelined Binary Tree Blocked": {"algo": pipelined_binary_tree_scan_blocked, "kwargs": {'b': b}},
    }

    results = []
    for data_type, local_data in inputs.items():
        for alg_name, algorithm in algorithms.items():
            try:
                time_it = []
                algo, algo_kwargs = algorithm["algo"], algorithm.get("kwargs", {})
                for i in range(5):
                    t, _ = benchmark_algorithm(algo, local_data, op, **algo_kwargs)
                    if i > 0:
                        time_it.append(t)
                avg_time = np.mean(time_it)
                if rank == 0:
                    results.append({
                        "block_size": b,
                        "data_type": data_type,
                        "algorithm": alg_name,
                        "time": avg_time,
                    })
                    print(f"Algorithm: {alg_name}, b {b}, Time: {avg_time:.6f} seconds")
                    sys.stdout.flush()
            except Exception as e:
                if rank == 0:
                    print(f"Error in {alg_name} for {data_type}: {e}, line: {e.__traceback__.tb_lineno}")
                    sys.stdout.flush()
                exit(1)

    if rank == 0:
        os.makedirs("results", exist_ok=True)
        existing_results = []
        if os.path.exists(json_file) and os.path.getsize(json_file) > 0:
            with open(json_file, "r") as f:
                existing_results = json.load(f)
        existing_results.extend(results)
        with open(json_file, "w") as f:
            json.dump(existing_results, f, indent=4)
    return results
    
if __name__ == "__main__":
    comm = MPI.COMM_WORLD   
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    input_size = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    json_file = sys.argv[2] if len(sys.argv) > 2 else "results/case2_results.json"
    
    # b = best_b_PBT(size, input_size)
    b = input_size // 14045  if input_size > 14045 else 1
    # b = 2
    # Algorithms to compare
    algorithms = {
        # "PLL": {"algo":PLL, "kwargs": {}},
        "Linear Pipeline": {"algo":linear_pipeline_scan, "kwargs": {'b': b}},
        "Doubly Pipelined Binary Tree": {"algo":doubly_pipelined_binary_tree_scan, "kwargs": {'b': b}},
        "Pipelined Binary Tree Blocked": {"algo":pipelined_binary_tree_scan_blocked, "kwargs": {'b': b}},
        # "Pipelined Binary Tree Blocked Overlap": {"algo":pipelined_binary_tree_scan_blkd_overlap, "kwargs": {'b': b}},
        "Pipelined Binary Tree": {"algo":pipelined_binary_tree_scan, "kwargs": {}},
        "Binomial Tree": {"algo":binomial_tree_scan, "kwargs": {}},
        "Simultaneous Binomial Tree": {"algo":simultaneous_binomial_tree_scan, "kwargs": {}},
        "Hypercube": {"algo":hypercube_scan, "kwargs": {}},
        "LLN": {"algo":lln, "kwargs": {}},
        "mpi_scan": {"algo": MPI.COMM_WORLD.scan, "kwargs": {}},
        "Recursive Doubling": {"algo":recursive_doubling_scan, "kwargs": {}},
    }
    # Define the binary operation 
    def add_op(x, y):
        return x + y

    def multiply_op(x, y):
        # return x * x / y * y
        return x * y  # Element-wise multiplication

    def matrix_multiply_op(x, y):
        return np.dot(x, y)
    # Benchmarking
   
    if len(sys.argv) == 2 or len(sys.argv) == 3:
        benchmark_victor_input(algorithms, multiply_op, input_size,json_file=json_file)
    elif len(sys.argv) == 4:
        b = int(sys.argv[3])
        benchmark_b( multiply_op, input_size, b, json_file=json_file)
    # else:
    #     benchmark_case1(algorithms, (comm, rank, size), add_op) 

    comm.Barrier()  # Ensure all processes finish before exiting
    if rank == 0:
        print("All benchmarks completed.")
        sys.stdout.flush()
    MPI.Finalize()
    
    

