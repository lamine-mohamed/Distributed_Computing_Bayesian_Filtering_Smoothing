import sys
from matplotlib import pyplot as plt
from mpi4py import MPI
import tensorflow as tf
from datetime import datetime
from scripts.ParallelKF_tf import *
from scripts.utils import configure_gpu, make_car_tracking_model
from scan_algorithms.Hybrid_scan import hybrid_scan
from scan_algorithms.simultaneous_binomial_tree import exclusive_simultaneous_binomial_tree_scan


# -------------------------------
# MPI Initialization
# -------------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# -------------------------------
# get command line arguments
# -------------------------------

data_size = int(sys.argv[1]) if len(sys.argv) > 1 else 100 # Default data size

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
