import copy
import numpy as np
from scripts.ParallelKF_tf import StateSpaceModel
from mpi4py import MPI

def make_car_tracking_model(q: float, dt: float, r: float, m0: np.ndarray, P0: np.ndarray):
    """
    Creates a simple constant-velocity car tracking state-space model.

    Parameters
    ----------
    q : float
        Process noise scaling factor.
    dt : float
        Time step size.
    r : float
        Measurement noise standard deviation.
    m0 : np.ndarray
        Initial state mean vector.
    P0 : np.ndarray
        Initial state covariance matrix.

    Returns
    -------
    StateSpaceModel
        Named tuple containing state-space matrices, initial state, and dimensions.
    """
    F = np.eye(4) + dt * np.eye(4, k=2)  # State transition matrix
    H = np.eye(2, 4)  # Observation matrix
    Q = np.kron(np.array([[dt**3/3, dt**2/2],
                          [dt**2/2, dt]]), np.eye(2))  # Process noise covariance
    R = r ** 2 * np.eye(2)  # Measurement noise covariance
    return StateSpaceModel(F, H, q * Q, R, m0, P0, m0.shape[0], H.shape[0])


def safe_copy(obj):
    """
    Makes a safe copy of an object using the best available method.

    Tries shallow copy first, then type constructor, then deep copy.

    Parameters
    ----------
    obj : any
        Object to copy.

    Returns
    -------
    copy of obj
    """
    try:
        return obj.copy()  # works for numpy arrays, lists, dicts
    except AttributeError:
        try:
            return type(obj)(obj)  # works for tuples, sets, basic types
        except Exception:
            return copy.deepcopy(obj)  # final fallback


def apply_op(x, y, op):
    """
    Applies an operation elementwise to two inputs, which can be nested tuples/lists or scalars.

    Parameters
    ----------
    x : scalar, list, or tuple
        First operand.
    y : scalar, list, or tuple
        Second operand.
    op : callable
        Binary operation to apply (e.g., operator.add, lambda a,b: a*b).

    Returns
    -------
    same type as x
        Result of applying op elementwise.
    """
    if isinstance(x, (tuple, list)):
        return type(x)(op(xi, yi) for xi, yi in zip(x, y))
    else:
        return op(x, y)
    

def benchmark_algorithm(algorithm, local_data, op, **kwargs):
    """
    Run a benchmark for a given scan algorithm.

    The function measures the wall-clock execution time of the provided
    `algorithm` on the current MPI process using a synchronized start
    and end (via `MPI.Barrier`).

    Parameters
    ----------
    algorithm : callable
        The scan algorithm function to benchmark. Must accept
        `(local_data, op, **kwargs)` as arguments.
    local_data : ndarray
        The local input array for this MPI process.
    op : callable
        Binary operation to use in the scan. Must be associative.
    **kwargs : dict, optional
        Additional keyword arguments passed to `algorithm`.

    Returns
    -------
    total_time : float
        The measured execution time in seconds.
    result : any
        The output returned by `algorithm` for this process.
    """

    comm = MPI.COMM_WORLD
    comm.Barrier()
    start_time = MPI.Wtime()
    result = algorithm(local_data, op, **kwargs)
    comm.Barrier()
    end_time = MPI.Wtime()
    total_time = end_time - start_time
    return total_time, result

def configure_gpu(memory_limit=1024, intra=2, inter=1):
    """
    Configure TensorFlow GPU memory usage and threading options.

    Parameters
    ----------
    memory_limit : int, optional
        Maximum GPU memory limit in MB (default: 1024 MB).
    intra : int, optional
        Number of intra-op parallelism threads (default: 2).
    inter : int, optional
        Number of inter-op parallelism threads (default: 1).

    Notes
    -----
    - Only applies if at least one GPU is available.
    - Must be called before TensorFlow allocates GPU memory.
    - Threading options apply to both CPU and GPU execution.
    """
    import tensorflow as tf

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
            )
        except RuntimeError as e:
            print(f"TensorFlow GPU configuration error: {e}")

    tf.config.threading.set_intra_op_parallelism_threads(intra)
    tf.config.threading.set_inter_op_parallelism_threads(inter)
