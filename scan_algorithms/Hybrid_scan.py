import tensorflow as tf
from mpi4py import MPI
from scripts.ParallelKF_tf import pkf, pks


@tf.function(reduce_retracing=True)
def update_filtered(local_result, offset, filtering_operator):
    """
    Apply an offset correction to local filtering results in parallel.

    This function broadcasts a per-rank filtering offset across 
    local results, then applies a filtering operator elementwise using
    TensorFlow's `vectorized_map`.

    Parameters
    ----------
    local_result : tuple of tf.Tensor
        Local filtering results `(A_local, b_local, C_local, J_local, eta_local)`.
    offset : tuple of tf.Tensor
        Single-step filtering offset `(A_offset, b_offset, C_offset, J_offset, eta_offset)`.
        Will be broadcast to the local shape before applying the operator.
    
    Returns
    -------
    tuple of tf.Tensor
        Local filtering results after applying the broadcasted offset.
    """
    A_local, b_local, C_local, J_local, eta_local = local_result
    A_offset, b_offset, C_offset, J_offset, eta_offset = offset

    A_offset_tiled = tf.broadcast_to(A_offset, tf.shape(A_local))
    b_offset_tiled = tf.broadcast_to(b_offset, tf.shape(b_local))
    C_offset_tiled = tf.broadcast_to(C_offset, tf.shape(C_local))
    J_offset_tiled = tf.broadcast_to(J_offset, tf.shape(J_local))
    eta_offset_tiled = tf.broadcast_to(eta_offset, tf.shape(eta_local))

    offset_tiled = (
        A_offset_tiled,
        b_offset_tiled,
        C_offset_tiled,
        J_offset_tiled,
        eta_offset_tiled
    )

    return tf.vectorized_map(
        filtering_operator,
        elems=(offset_tiled, (A_local, b_local, C_local, J_local, eta_local)),
        fallback_to_while_loop=False
    )


@tf.function(reduce_retracing=True)
def update_smoothed(local_result, offset, smoothing_operator):
    """
    Apply an offset correction to local smoothing results in parallel.

    This function broadcasts a per-rank smoothing offset across 
    local results, then applies a smoothing operator elementwise using
    TensorFlow's `vectorized_map`.

    Parameters
    ----------
    local_result : tuple of tf.Tensor
        Local smoothing results `(E_local, g_local, L_local)`.
    offset : tuple of tf.Tensor
        Single-step smoothing offset `(E_offset, g_offset, L_offset)`.
        Will be broadcast to the local shape before applying the operator.
    
    Returns
    -------
    tuple of tf.Tensor
        BLocal smoothing results after applying the broadcasted offset.
    """
    E_local, g_local, L_local = local_result
    E_offset, g_offset, L_offset = offset

    E_offset_tiled = tf.broadcast_to(E_offset, tf.shape(E_local))
    g_offset_tiled = tf.broadcast_to(g_offset, tf.shape(g_local))
    L_offset_tiled = tf.broadcast_to(L_offset, tf.shape(L_local))

    offset_tiled = [E_offset_tiled, g_offset_tiled, L_offset_tiled]

    return tf.vectorized_map(
        smoothing_operator,
        elems=(offset_tiled, (E_local, g_local, L_local)),
        fallback_to_while_loop=False
    )


def hybrid_scan(algorithm, local_data, Fop, Sop, model):
    """
    Perform a hybrid MPI + GPU/CPU scan for distributed Kalman filtering and smoothing.

    This procedure runs a parallel scan across MPI ranks for both the
    filtering and smoothing passes of a state-space model.  
    Each MPI process computes local filtering results, exchanges offsets
    with neighbors using `algorithm`, and applies offset corrections using
    GPU/CPU-accelerated TensorFlow operations.

    Parameters
    ----------
    algorithm : callable
        MPI-compatible scan function for computing per-rank offsets.
        Must accept `(data, operator, comm=...)` or `(data, operator)` arguments.
    local_data : tuple of tf.Tensor
        Observations or intermediate state needed for local Kalman filtering.
    Fop : callable
        Filtering operator for combining two sets of filtering results.
    Sop : callable
        Smoothing operator for combining two sets of smoothing results.
    model : object
        State-space model definition required by `pkf` and `pks`.

    Returns
    -------
    runtime : float
        Total wall-clock time for the distributed scan in seconds.
    filtered : tuple of tf.Tensor
        Filtered results after applying distributed offsets.
    result : tuple of tf.Tensor
        Smoothed results after applying distributed offsets.

    Notes
    -----
    The algorithm runs in **four main steps**:

    1. **Local Filtering (GPU/CPU)**:
       - Each rank runs `pkf` to compute partial filtering results from its local data.

    2. **Filtering Offset Scan (MPI)**:
       - Each rank extracts its last local result as an offset.
       - `algorithm` is used to scan offsets across ranks.
       - Non-root ranks apply `update_filtered` to incorporate the offset.

    3. **Local Smoothing (GPU/CPU)**:
       - Each rank runs `pks` to compute partial smoothing results from filtered data.

    4. **Smoothing Offset Scan (MPI)**:
       - Rank order is reversed to propagate backward offsets.
       - `algorithm` or `comm.exscan` is used to compute backward offsets.
       - Non-last ranks apply `update_smoothed` to incorporate the offset.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    comm.Barrier()
    start_time = MPI.Wtime()

    filtered = pkf(model, local_data, first_rank=(rank == 0))

    offset = [subelement[-1] for subelement in filtered]
    comm.Barrier()
    offset = algorithm(offset, Fop)

    comm.Barrier()
    if rank != 0:
        filtered = update_filtered(filtered, offset, Fop)

    smoothed = pks(model, filtered[1], filtered[2], last_rank=(rank == size - 1))

    offset = [subelement[0] for subelement in smoothed]
    new_order = list(reversed(range(size)))
    newComm = comm.Create(comm.Get_group().Incl(new_order))

    offset = algorithm(offset, Sop, comm=newComm) if algorithm != comm.exscan else newComm.exscan(offset, Sop)

    if rank != size - 1:
        smoothed = update_smoothed(smoothed, offset, Sop)

    comm.Barrier()
    end_time = MPI.Wtime()

    runtime = end_time - start_time

    return runtime, filtered, smoothed
