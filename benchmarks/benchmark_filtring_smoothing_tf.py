import argparse
from datetime import datetime
import gc
import json
import os
import sys
import time

from matplotlib import pyplot as plt
from scripts.ParallelKF_tf import *
from scan_algorithms.lln import exclusive_LLN
from scan_algorithms.binomial_tree import binomial_tree_exclusive_scan
from scan_algorithms.simultaneous_binomial_tree import exclusive_simultaneous_binomial_tree_scan
from scan_algorithms.Pipelined_Binary_Tree import exclusive_pipelined_binary_tree_scan
from scan_algorithms.Hypercub_scan import Hypercube_Scan_exclusive
from scan_algorithms.Linear_Pipeline import exclusive_linear_scan
from scripts.utils import make_car_tracking_model
from scripts.utils import configure_gpu

from mpi4py import MPI

from scan_algorithms.Hybrid_scan import hybrid_scan


def benchmark_different_algos(data_size: int, num_trials: int = 10):
    """
    Benchmark various scan algorithms using distributed MPI data on various data sizes.

    parameters
    ----------
    data_size : int
        Number of data elements to use for the benchmark.
    num_trials : int, optional
        Number of trials to run for each algorithm (default is 10).
    Returns
    ------- 
    None        
    """
    if data_size == 0:
        print(f"Rank {rank} has no elements to process", flush=True)
        return

    # Define algorithms to benchmark
    algos = {
        "excluive_pipelined_binary_tree_scan": exclusive_pipelined_binary_tree_scan,
        "exclusive_LLN": exclusive_LLN,
        "MPI_exscan": comm.exscan,
        "exclusive_simultaneous_binomial_tree_scan": exclusive_simultaneous_binomial_tree_scan,
        "binomial_tree_exclusive_scan": binomial_tree_exclusive_scan,
        "Hypercube_Scan_exclusive": Hypercube_Scan_exclusive,
        "ex_linear_scan": exclusive_linear_scan,
    }

    empirical_results = {name: [] for name in algos}
    logT = int(np.log10(data_size))
    input_sizes = np.logspace(2, logT, num=10, base=10, dtype=int)

    if rank == 0:
        print(f"Rank {rank} logT: {logT}, data_size: {data_size}, input_sizes: {input_sizes}", flush=True)

    filter_op = FilteringOperator(dim=car_tracking_model.xdim)
    smooth_op = SmoothingOperator(dim=car_tracking_model.xdim)

    for obs_size in input_sizes:
        if rank == 0:
            observations = np.load("data/car_tracking/observations.npz")["ys"][:obs_size]
            print(f"Rank {rank} loaded observations with shape {observations.shape}", flush=True)
            sendbuf = np.array_split(observations, size)
        else:
            sendbuf = None

        for algo_name, algo in algos.items():
            if rank == 0:
                print(f"Rank {rank} starting benchmark for {algo_name}", flush=True)

            local_data = comm.scatter(sendbuf, root=0)
            times = []

            for trial in range(num_trials + 1):
                exec_time, _, _ = hybrid_scan(
                    algorithm=algo,
                    local_data=local_data,
                    Fop=filter_op,
                    Sop=smooth_op,
                    model=car_tracking_model
                )
                if trial > 1:  # Skip first trial for warm-up
                    times.append(exec_time)

            avg_time = np.mean(times)
            empirical_results[algo_name].append(avg_time)

            if rank == 0:
                print(f"Rank {rank} finished {algo_name} for data size {obs_size}, time: {avg_time:.4f}s", flush=True)

    # Plot results
    if rank == 0:
        for algo_name, times in empirical_results.items():
            plt.plot(input_sizes[:len(times)]*size, times, marker='o', label=algo_name)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Data Size (log scale)')
        plt.ylabel('Execution Time (s)')
        plt.title('Benchmark of Hybrid Associative Filtering')
        plt.grid(which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/car_tracking_benchmark_Hybrid_{size}_ranks_D{data_size}_{current_time}.svg", format='svg')
        plt.show()



def benchmark_shared_distributed(data_size: int, num_trials: int = 5):
    """
    Benchmark hybrid and local scans on various data sizes using shared and distributed memory.

    parameters  
    ----------
    data_size : int
        Number of data elements to use for the benchmark.
    num_trials : int, optional
        Number of trials to run for each algorithm (default is 5).
    Returns
    -------
    None
    """
    if data_size == 0:
        print(f"Rank {rank} has no elements to process", flush=True)
        return

    empirical_results = {"hybrid_scan": [], "local_scan": []}
    logT = np.log10(data_size)
    input_sizes = np.logspace(np.log10(size*10), logT, num=10, base=10, dtype=int)

    filter_op = FilteringOperator(dim=car_tracking_model.xdim)
    smooth_op = SmoothingOperator(dim=car_tracking_model.xdim)

    # Local (CPU/GPU) scan benchmark
    for obs_size in input_sizes:
        if rank == 0:
            observations = np.load("data/car_tracking/observations.npz")["ys"][:obs_size]
            times = []
            for trial in range(num_trials + 1):
                start_t = time.time()
                _ = pkfs(car_tracking_model, observations, max_parallel=int(np.ceil(obs_size)))
                exec_time = time.time() - start_t
                if trial > 1:
                    times.append(exec_time)
            empirical_results["local_scan"].append(np.mean(times))
            del observations

    # Free GPU memory
    comm.Barrier()
    tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()
    gc.collect()

    # Distributed hybrid scan benchmark
    for obs_size in input_sizes:
        if rank == 0:
            observations = np.load("data/car_tracking/observations.npz")["ys"][:obs_size]
            sendbuf = np.array_split(observations, size)
        else:
            sendbuf = None

        local_data = comm.scatter(sendbuf, root=0)
        times = []
        for trial in range(num_trials + 1):
            exec_time, _, _ = hybrid_scan(
                algorithm=binomial_tree_exclusive_scan,
                local_data=local_data,
                Fop=filter_op,
                Sop=smooth_op,
                model=car_tracking_model
            )
            if trial > 1:
                times.append(exec_time)
        empirical_results["hybrid_scan"].append(np.mean(times))

    if rank == 0:
        # Plot results
        for algo_name, times in empirical_results.items():
            plt.plot(input_sizes[:len(times)], times, marker='o', label=algo_name)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Data Size (log scale)')
        plt.ylabel('Execution Time (seconds)')
        plt.title('Benchmark of Hybrid Associative Filtering')
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.legend()
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/car_tracking_benchmark_Hybrid_{size}_ranks_D{data_size}_{current_time}.svg", format='svg')
        plt.show()



def benchmark_across_mpi_sizes_fixed_data(data_size, num_trials=5):
    """
    Benchmark various scan algorithms across different MPI sizes using a fixed dataset.

    Parameters
    ----------
    data_size : int
        Total number of observations to process across all MPI ranks.
        If `0`, the function exits early.
    num_trials : int, optional
        Number of timed trials to run for each algorithm (excluding warm-up).
        Default is 5.

    Notes
    -----
    - The first trial for each algorithm is a warm-up and not included in the average.
    - The results are appended to ``../results/case1_results.json``. Existing results are preserved.
    - The ``exclusive_LLN`` algorithm is skipped if the MPI size is less than 10.

    Side Effects
    ------------
    Writes merged benchmark results to ``../results/case1_results.json``.

    """
    if data_size == 0:
        if rank == 0:
            print("Data size is zero. Exiting.")
        return

    algos = {
        "excluive_pipelined_binary_tree_scan": exclusive_pipelined_binary_tree_scan,
        "exclusive_LLN": exclusive_LLN,
        "MPI_exscan": comm.exscan,
        "exclusive_simultaneous_binomial_tree_scan": exclusive_simultaneous_binomial_tree_scan,
        "binomial_tree_exclusive_scan": binomial_tree_exclusive_scan,
        "Hypercube_Scan_exclusive": Hypercube_Scan_exclusive,
        "ex_linear_scan": exclusive_linear_scan,
    }

    empirical_results = {name: [] for name in algos}

    # Scatter fixed dataset across ranks
    if rank == 0:
        observations = np.load("data/car_tracking/observations.npz")["ys"][:data_size]
        print(f"[Rank {rank}] Loaded fixed dataset with {data_size} observations.", flush=True)
        sendbuf = np.array_split(observations, size)
    else:
        sendbuf = None

    local_data = comm.scatter(sendbuf, root=0)

    filter_OP = FilteringOperator(dim=car_tracking_model.xdim)
    smooth_OP = SmoothingOperator(dim=car_tracking_model.xdim)

    # Benchmark each algorithm
    for algo_name, algo in algos.items():
        if algo_name == "exclusive_LLN" and size < 10:
            if rank == 0:
                print(f"[Rank {rank}] Skipping {algo_name} for size {size} ranks "
                      f"(requires at least 10 ranks)", flush=True)
            empirical_results[algo_name].append(0)
            continue

        if rank == 0:
            print(f"[Rank {rank}] Starting benchmark for algorithm: {algo_name}", flush=True)

        times = []
        for trial in range(num_trials + 1):  # +1 for warm-up
            exec_time, _, _ = hybrid_scan(
                algorithm=algo,
                local_data=local_data,
                Fop=filter_OP,
                Sop=smooth_OP,
                model=car_tracking_model
            )
            if trial > 0:  # Skip warm-up
                times.append(exec_time)

        avg_time = np.mean(times)
        empirical_results[algo_name].append(avg_time)

        if rank == 0:
            print(f"[Rank {rank}] Algorithm: {algo_name} with {size} ranks took {avg_time:.4f}s",
                  flush=True)

    # Save results on rank 0
    if rank == 0:
        results_path = "../results/case1_results.json"

        if os.path.exists(results_path) and os.path.getsize(results_path) > 0:
            with open(results_path, "r") as f:
                existing_results = json.load(f)
        else:
            existing_results = {}

        for algo_name, new_times in empirical_results.items():
            existing_results.setdefault(algo_name, []).extend(new_times)

        with open(results_path, "w") as f:
            json.dump(existing_results, f, indent=4)







def parse_args():
    parser = argparse.ArgumentParser(description="Car Tracking Benchmark Runner")
    parser.add_argument(
        "benchmark", type=str, nargs="?", default="shared_distributed",
        choices=["different_algos", "mpi_fixed_data", "shared_distributed", "hybrid_scan"],
        help="Benchmark to run (default: shared_distributed)"
    )
    parser.add_argument(
        "data_size", type=int, nargs="?", default=1000,
        help="Number of data elements to use for the benchmark (default: 1000)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    """
    MPI entrypoint for benchmarking various filtering and smoothing algorithms.

    This script:
    1. Initializes MPI communication (rank, size).
    2. Parses command-line arguments to determine the benchmark to run.
    3. Configures GPU memory and threading options.
    4. Creates a car-tracking state-space model.
    5. Dispatches execution to the requested benchmark function.

    Command-line arguments (parsed via parse_args()):
    -------------------------------------------------
    --data_size   (int)   Number of observations to use.
    --benchmark   (str)   Benchmark type:
        - "different_algos"     → Compare different algorithms.
        - "mpi_fixed_data"      → Fixed dataset across varying MPI sizes.
        - "shared_distributed"  → Shared-distributed benchmark.
        - "hybrid_scan"         → Hybrid scan benchmark (plots results).

    Example:
    --------
    mpirun -np 4 python3 benchmarks/benchmark_filtring_smoothing_tf.py \
        --data_size 500 --benchmark mpi_fixed_data
    """

    # -------------------------------
    # MPI Initialization
    # -------------------------------
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # -------------------------------
    # Parse CLI arguments
    # -------------------------------
    args = parse_args()
    data_size = args.data_size
    benchmark_name = args.benchmark

    # -------------------------------
    # Configure GPU for TensorFlow
    # -------------------------------
    configure_gpu(memory_limit=1024)  # Limit GPU to 1024 MB

    # -------------------------------
    # Build the car tracking model
    # -------------------------------
    car_tracking_model = make_car_tracking_model(
        q=1.0, dt=0.1, r=0.5,
        m0=np.array([0.0, 0.0, 1.0, -1.0]),
        P0=np.eye(4)
    )

    if rank == 0:
        print(f"Rank {rank} starting benchmark '{benchmark_name}' "
              f"with {data_size} elements", flush=True)

    # -------------------------------
    # Benchmark Selection
    # -------------------------------
    with tf.device('/GPU:0'):
        if benchmark_name == "different_algos":
            benchmark_different_algos(data_size, num_trials=10)

        elif benchmark_name == "mpi_fixed_data":
            benchmark_across_mpi_sizes_fixed_data(data_size, num_trials=10)

        elif benchmark_name == "shared_distributed":
            benchmark_shared_distributed(data_size, num_trials=5)

        elif benchmark_name == "hybrid_scan":
            # Switch to CPU for hybrid scan
            with tf.device('/CPU:0'):
                if rank == 0:
                    # Load dataset
                    observations = np.load("data/car_tracking/observations.npz")["ys"]
                    true_states = np.load("data/car_tracking/true_states.npz")["true_xs"]

                    if data_size is not None:
                        observations = observations[:data_size]

                    print(f"Rank {rank} loaded observations with shape "
                          f"{observations.shape}", flush=True)

                    # Split data across MPI ranks
                    sendbuf = np.array_split(observations, size)
                else:
                    sendbuf = None

                local_data = comm.scatter(sendbuf, root=0)

            # Create filtering and smoothing operators
            Fop = FilteringOperator(dim=car_tracking_model.xdim)
            Sop = SmoothingOperator(dim=car_tracking_model.xdim)

            # Run hybrid scan benchmark
            exec_time, global_filtered, global_smoothed = hybrid_scan(
                exclusive_simultaneous_binomial_tree_scan,
                local_data=local_data,
                Fop=Fop,
                Sop=Sop,
                model=car_tracking_model
            )

            # -------------------------------
            # Gather and visualize results
            # -------------------------------
            global_filtered = comm.gather(global_filtered, root=0)
            global_smoothed = comm.gather(global_smoothed, root=0)

            if rank == 0:
                print(f"Hybrid scan took {exec_time:.4f} seconds")

                # Concatenate gathered results
                global_filtered = [tf.concat(g, axis=0) for g in zip(*global_filtered)]
                global_smoothed = [tf.concat(g, axis=0) for g in zip(*global_smoothed)]

                pfms = global_filtered[1].numpy()
                sms = global_smoothed[1].numpy()

                # Plot first 100 timesteps
                plt.figure(figsize=(7, 7))
                plt.plot(pfms[:100, 0], pfms[:100, 1],
                         label="Filtered", color="g", linestyle="--", linewidth=2)
                plt.plot(sms[:100, 0], sms[:100, 1],
                         label="Smoothed", color="b", linestyle="-", linewidth=2)
                plt.plot(observations[:100, 0], observations[:100, 1],
                         'r.', label='Observations', markersize=10)
                plt.plot(true_states[:100, 0], true_states[:100, 1],
                         'k-', label='True State', linewidth=2)

                plt.xlabel('x', fontsize=12)
                plt.ylabel('y', fontsize=12)
                plt.legend()

                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                plt.savefig(f"plots/car_tracking_Hybrid_{size}_ranks_D{data_size}_{current_time}.svg",
                            format='svg')
                plt.show()
