from itertools import cycle
import subprocess
import sys
import time
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import numpy as np
import tensorflow as tf
# path to benchmarks
def plot_results(file_name, x_col, y_col, title, xlabel, ylabel, difMPIsize = False, **kwargs):
    with open(file_name, "r") as f:
        data = json.load(f)
    # Group by algorithm and data type
    if not difMPIsize:
        df = pd.DataFrame(data)
        grouped = df.groupby(["algorithm", "data_type"])
        plt.figure(figsize=(10, 6))  # Create a single figure
        for (algorithm, data_type), group in grouped:
            plt.plot(group[x_col], group[y_col], marker='o', label=f"{algorithm}_{data_type}")
    
    else:  
        # grouped = df.groupby("algorithm")
        # load all algorithms in one plot
        max_Wsize = kwargs.get('max_Wsize', 16)  # Default max W size
        plt.figure(figsize=(10, 6))  # Create a single figure
        for algo_name, times in data.items():
            plt.plot(range(2,max_Wsize+1,2), times, marker='o', label=algo_name)
    # plt.xscale('log', base=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    # plt.grid()
    plt.xscale('log', base=10)  # Set x-axis to logarithmic scale
    plt.yscale('log', base=10)  # Set y-axis to logarithmic scale
    plt.grid(which='both', linestyle='--', linewidth=0.5)  # Add grid lines
    plt.legend()  # Add a legend for all lines
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f"plots/{title.replace(' ', '_').lower()}_{current_time}.svg", format='svg')
    plt.show()
    # plt.close()

# def plot_results3(file_name, x_col, y_col, z_col, title, xlabel, ylabel, zlabel):
#     with open(file_name, "r") as f:
#         data = json.load(f)
#     df = pd.DataFrame(data)
#     # Group by algorithm, data type, and mpi_size
#     grouped = df.groupby(["algorithm", "data_type", x_col])
#     color_cycle = cycle(plt.cm.tab10.colors)
#     plt.figure(figsize=(10, 6))
    
#     # Define marker styles for different mpi_sizes
#     markers = ['o', 's', 'H', 'D', '+', '*', '>', 'p', '*', 'h', 'D', '+', 'x', 'X', 'd']
    
#     # Track colors for each algorithm
#     algorithm_colors = {}
    
#     for (algorithm, data_type, mpi_size), group in grouped:
#         # Assign a consistent color for each algorithm
#         if algorithm not in algorithm_colors:
#             algorithm_colors[algorithm] = next(color_cycle)  # Use a colormap for colors
        
#         # Select a marker based on mpi_size (cycling through markers if needed)
#         marker = markers[mpi_size % len(markers)]
        
#         # Plot the data
#         plt.plot(
#             group[y_col],
#             group[z_col],
#             marker=marker,
#             label=f"{algorithm}_{data_type}_MPI={mpi_size}",
#             color=algorithm_colors[algorithm]
#         )
    
#     plt.xscale('log', base=2)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.title(title)
#     plt.grid()
#     plt.legend()  # Add a legend for all lines
#     plt.savefig(f"plots/{title.replace(' ', '_').lower()}.png")
#     plt.show()

if __name__ == "__main__":
    # plot_results("results/case1_results.json", "mpi_size", "time", f"Case 1: Varying MPI size; data size: {10000} ", "MPI size", "Time (s)", difMPIsize=True, max_Wsize=48)

    # sys.exit(0)
    benchmark = sys.argv[1] if len(sys.argv) > 1 else "all"
    benchmarks = ["case1", "case2", "case3", "case4", "all"]
    if benchmark not in benchmarks:
        print(f"Invalid benchmark specified. Choose from {benchmarks}.")
        sys.exit(1)
    
    def case1():
        print("GPUs:", tf.config.list_physical_devices('GPU'))
        
        max_Wsize = int(sys.argv[2]) if len(sys.argv) > 2 else 16  # Default max W size
        data_size = int(sys.argv[3]) if len(sys.argv) > 3 else 10000 # Default data size
        #  scale mpi size
        mpi_sizes = range(2,max_Wsize+1,2) # 10^1 to 10^6
        print("Running Case 1 (Varying MPI size)...")
        os.makedirs("results", exist_ok=True)
        open("results/case1_results.json", "w").close()  # Clear previous
        for p in mpi_sizes:
            print(f"Running with {p} MPI processes...")
            sys.stdout.flush()
            # Run the benchmark script with varying MPI sizes
            try:
                # Run the benchmark script with varying MPI sizes
                subprocess.run(
                    ["mpirun", "-np", str(p), "--oversubscribe",  "python3", "benchmarks/benchmark_filtring_smoothing_tf.py", str(data_size), "results/case1_results.json"],
                    check=True
                )
            except subprocess.CalledProcessError as e:
                print(f"Error: Command failed with exit code {e.returncode}")
                print(f"Command: {e.cmd}")
                break  # Stop execution if a command fails

        # plot results case 1
        plot_results("results/case1_results.json", "mpi_size", "time", f"Case 1: Varying MPI size; data size: {data_size} ", "MPI size", "Time (s)", difMPIsize=True, max_Wsize=max_Wsize)
        
    def case2():
        # input_sizes = range(10, 10**9, 10**3)  
        mpi_size = sys.argv[2] if len(sys.argv) > 2 else 20  # Default MPI size
        maxlog = int(sys.argv[3]) if len(sys.argv) > 3 else 22  # Default max log size
        input_sizes = np.logspace(8, maxlog, num=10, base=2).astype(int)
        os.makedirs("results", exist_ok=True)
        # Clear previous results
        open("results/case2_results.json", "w").close()
        for input_size in input_sizes:
            print(f"Running with input size 2**{int(np.log2(input_size))} = {input_size}...")
            # Run the benchmark script with varying input sizes
            try:
                # Run the benchmark script with varying input sizes
                subprocess.run(
                    ["mpirun", "-np", str(mpi_size), "--oversubscribe", "python3", "benchmarks/run_benchmarks.py", str(input_size)],
                    check=True
                )
                time.sleep(1)  # Optional: add a small delay between runs
            except subprocess.CalledProcessError as e:
                print(f"Error: Command failed with exit code {e.returncode}")
                print(f"Command: {e.cmd}")
                break  # Stop execution if a command fails

        # plote results case 2
        title = f"Execution Time vs Input Size (mpi size={mpi_size}, data type=double)"
        plot_results("results/case2_results.json", "input_size", "time", title, "input size", "Time (s)")
    
    # def case3():
    #     # mpi_sizes = [2**i for i in range(1, 6)]  # MPI sizes: 2^1 to 2^5
    #     mpi_sizes = [20, 30, 40, 50, 60, 80]  # MPI sizes: 2 to 32
    #     input_sizes = [2**i for i in range(10, 28)]  # Input sizes: 2^10 to 2^19
    #     print("Running Case 3 (Varying MPI size and input size)...")
    #     os.makedirs("results", exist_ok=True)
    #     # Clear previous results
    #     json_file = "results/case3_results.json"
    #     open(json_file, "w").close()
        
    #     for mpi_size in mpi_sizes:
    #         for input_size in input_sizes:
    #             print(f"Running with MPI size {mpi_size} and input size {input_size}...")
    #             try:
    #                 # Run the benchmark script with varying MPI sizes and input sizes
    #                 subprocess.run(
    #                     ["mpirun", "-np", str(mpi_size), "python3", "benchmarks/run_benchmarks.py", str(input_size), json_file ],
    #                     check=True
    #                 )
    #                 # time.sleep(1)  # Optional: add a small delay between runs
    #             except subprocess.CalledProcessError as e:
    #                 print(f"Error: Command failed with exit code {e.returncode}")
    #                 print(f"Command: {e.cmd}")
    #                 break  # Stop execution if a command fails

    #         # plot_results(
    #         #     json_file,
    #         #     #inpute sizes coresspond to current mpi size
    #         #     ,
    #         #     "time",
    #         #     f"Case 3: Varying MPI size {mpi_size} and input size",
    #         #     "MPI size", "Input size",
    #         #     "Time (s)"
    #         # )

    #     # Plot results for case 3
    #     plot_results3(
    #         json_file,
    #         "mpi_size","input_size",
    #         "time",
    #         "Case 3: Varying MPI size and input size",
    #         "MPI size", "Input size",
    #         "Time (s)"
    #     )


    def case4():
        # benchmark b 
        log2T = 12
        input_size = 2**log2T  
        b = np.logspace(2, log2T, num=100, base=2).astype(int)
        json_file = "results/case4_results.json"
        print("Running Case 4 (Varying block size)...")
        os.makedirs("results", exist_ok=True)
        # Clear previous results
        open(json_file, "w").close()
        for block_size in b:
            print(f"Running with block size {block_size}...")
            try:
                # Run the benchmark script with varying block sizes
                subprocess.run(
                    ["mpirun", "-np", "16", "python3", "benchmarks/run_benchmarks.py", str(input_size), json_file, str(block_size)],
                    check=True
                )
                # time.sleep(1)  # Optional: add a small delay between runs
            except subprocess.CalledProcessError as e:
                print(f"Error: Command failed with exit code {e.returncode}")
                print(f"Command: {e.cmd}")
                break
        
        plot_results(
            json_file,
            "block_size",
            "time",
            "Case 4: Varying block size",
            "Block size", "Time (s)"
        )

    if benchmark == "case1":
        case1()
    elif benchmark == "case2":
        case2()
    elif benchmark == "case3":
        # case3()
        print("Case 3 is not implemented yet. Please run case1 or case2.")
    elif benchmark == "case4":
        case4()
    else:   
        case1()
        case2()


print(":white_check_mark: All done! Plots saved to: results/plots/")
