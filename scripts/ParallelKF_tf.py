"""
Parallel Kalman Filtering and Smoothing Module
----------------------------------------------

This module implements parallelized Kalman filtering (PKF) and parallelized Kalman smoothing (PKS)
using TensorFlow and TensorFlow Probability. It supports batching and efficient associative scans
for both filtering and smoothing operations. MPI is optionally imported for distributed applications.

Dependencies:
    - TensorFlow
    - TensorFlow Probability
    - NumPy
"""

from collections import namedtuple
import math

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

# Aliases for common matrix operations
mm = tf.linalg.matmul  # matrix-matrix multiplication
mv = tf.linalg.matvec  # matrix-vector multiplication

# ----------------------------
# State-Space Model Definition
# ----------------------------

StateSpaceModel = namedtuple(
    "StateSpaceModel",
    ["F", "H", "Q", "R", "m0", "P0", "xdim", "ydim"]
)

# ----------------------------
# Filtering Element Functions
# ----------------------------

@tf.function(reduce_retracing=True)
def first_filtering_element(model, y):
    """
    Computes the first filtering element for parallel Kalman filtering.

    Parameters
    ----------
    model : StateSpaceModel
        The state-space model.
    y : tf.Tensor or np.ndarray
        Observation vector at first time step.

    Returns
    -------
    list of tf.Tensor
        A list of elements [A, b, C, J, eta] suitable for associative scan.
    """
    m1 = mv(model.F, model.m0)
    P1 = model.F @ mm(model.P0, model.F, transpose_b=True) + model.Q
    S1 = model.H @ mm(P1, model.H, transpose_b=True) + model.R
    S1_chol = tf.linalg.cholesky(S1)
    K1t = tf.linalg.cholesky_solve(S1_chol, model.H @ P1)

    A = tf.zeros_like(model.F)
    b = m1 + mv(K1t, y - mv(model.H, m1), transpose_a=True)
    C = P1 - mm(K1t, S1, transpose_a=True) @ K1t

    HF = model.H @ model.F
    S = model.H @ mm(model.Q, model.H, transpose_b=True) + model.R
    chol = tf.linalg.cholesky(S)
    eta = mv(HF, tf.squeeze(tf.linalg.cholesky_solve(chol, tf.expand_dims(y, 1)), 1), transpose_a=True)
    J = mm(HF, tf.linalg.cholesky_solve(chol, HF), transpose_a=True)

    return [A, b, C, J, eta]

@tf.function(reduce_retracing=True)
def generic_filtering_element(model, y):
    """
    Computes a generic filtering element for parallel Kalman filtering.

    Parameters
    ----------
    model : StateSpaceModel
        The state-space model.
    y : tf.Tensor or np.ndarray
        Array of Observation vectors.

    Returns
    -------
    list of tf.Tensor
        A list of elements [A, b, C, J, eta] suitable for associative scan.
    """
    S = model.H @ mm(model.Q, model.H, transpose_b=True) + model.R
    chol = tf.linalg.cholesky(S)
    Kt = tf.linalg.cholesky_solve(chol, model.H @ model.Q)

    A = model.F - mm(Kt, model.H, transpose_a=True) @ model.F
    b = mv(Kt, y, transpose_a=True)
    C = model.Q - mm(Kt, model.H, transpose_a=True) @ model.Q

    HF = model.H @ model.F
    eta = mv(HF, tf.squeeze(tf.linalg.cholesky_solve(chol, tf.expand_dims(y, 1)), 1), transpose_a=True)
    J = mm(HF, tf.linalg.cholesky_solve(chol, HF), transpose_a=True)
    return [A, b, C, J, eta]

@tf.function(reduce_retracing=True)
def make_associative_filtering_elements(model, observations, first_rank=False):
    """
    Converts observations into associative filtering elements for parallel scan.

    Parameters
    ----------
    model : StateSpaceModel
        The state-space model.
    observations : tf.Tensor or np.ndarray
        Sequence of observation vectors.
    first_rank : bool, optional
        True if the first observation requires special handling (default False).

    Returns
    -------
    tuple of tf.Tensor
        Tuple of tensors representing associative filtering elements.
    """
    if first_rank:
        first_elems = first_filtering_element(model, observations[0])
        generic_elems = tf.vectorized_map(
            lambda o: generic_filtering_element(model, o),
            observations[1:],
            fallback_to_while_loop=False
        )
        return tuple(tf.concat([tf.expand_dims(first_e, 0), gen_es], 0)
                     for first_e, gen_es in zip(first_elems, generic_elems))
    else:
        generic_elems = tf.vectorized_map(
            lambda o: generic_filtering_element(model, o),
            observations,
            fallback_to_while_loop=False
        )
        return tuple(generic_elems)

# ----------------------------
# Filtering Operator
# ----------------------------

@tf.function(reduce_retracing=True)
def filtering_operator(elems):
    """
    Associative operator to combine two filtering elements for PKF.

    Parameters
    ----------
    elems : tuple of list of tf.Tensor
        A tuple containing two filtering elements.

    Returns
    -------
    list of tf.Tensor
        The combined filtering element.
    """
    elem1, elem2 = elems
    A1, b1, C1, J1, eta1 = elem1
    A2, b2, C2, J2, eta2 = elem2
    dim = A1.shape[0]
    I = tf.eye(dim, dtype=A1.dtype)

    temp = tf.linalg.solve(I + C1 @ J2, tf.transpose(A2), adjoint=True)
    A = mm(temp, A1, transpose_a=True)
    b = mv(temp, b1 + mv(C1, eta2), transpose_a=True) + b2
    C = mm(temp, mm(C1, A2, transpose_b=True), transpose_a=True) + C2

    temp = tf.linalg.solve(I + J2 @ C1, A1, adjoint=True)
    eta = mv(temp, eta2 - mv(J2, b1), transpose_a=True) + eta1
    J = mm(temp, J2 @ A1, transpose_a=True) + J1

    return [A, b, C, J, eta]

class FilteringOperator:
    """
    Wrapper class for filtering_operator with identity initialization.

    Attributes
    ----------
    dim : int
        State dimension.
    identity : list of tf.Tensor
        Identity filtering element [A, b, C, J, eta].
    """

    def __init__(self, dim):
        """
        Initialize the filtering operator with identity element.

        Parameters
        ----------
        dim : int
            State dimension.
        """
        self.dim = dim
        self.identity = self.init_identity()

    @tf.function(reduce_retracing=True)
    def __call__(self, *args):
        """
        Apply the filtering operator on call.

        Parameters
        ----------
        *args : list of tf.Tensor
            A tuple of two filtering elements or two separate filtering elements.

        Returns
        -------
        list of tf.Tensor
            Combined filtering element.
        """
        if len(args) == 1:
            return filtering_operator(args[0])
        elif len(args) == 2:
            return filtering_operator((args[0], args[1]))
        else:
            raise ValueError(f"Expected 1 or 2 arguments, got {len(args)}")

    def init_identity(self):
        """
        Create identity filtering element.

        Returns
        -------
        list of tf.Tensor
            Identity element [A, b, C, J, eta].
        """
        I = tf.eye(self.dim, dtype=tf.float64)
        Z = tf.zeros((self.dim,), dtype=tf.float64)
        ZM = tf.zeros((self.dim, self.dim), dtype=tf.float64)
        return [I, Z, ZM, ZM, Z]  # [A, b, C, J, eta]


@tf.function(reduce_retracing=True)
def pkf(model, observations, max_parallel=100000, first_rank=False, initialize=True):
    """
    Parallel Kalman Filter (PKF) using an associative scan.

    This function performs parallel filtering on a sequence of observations 
    for a given state-space model, leveraging TensorFlow Probability's 
    associative scan to enable high parallelism.

    Parameters
    ----------
    model : StateSpaceModel
        The state-space model containing matrices (F, Q, H, R) and initial state.
    observations : tf.Tensor or np.ndarray
        Observations over time.
    max_parallel : int, optional
        Maximum number of parallel operations in the associative scan. 
        Default is 100,000.
    first_rank : bool, optional
        Whether to treat the first element specially (affects scan initialization). 
        Default is False.
    initialize : bool, optional
        If True, initializes filtering element using the model and observation.

    Returns
    -------
    tuple of tf.Tensor
        The final elements of the parallel scan, typically containing tuples of:
        - A : Transition-related matrices
        - b : Filtered state means
        - C : Filtered state covariances
        - J : Auxiliary matrices used in the scan
        - eta : Auxiliary vectors used in the scan
    """
    if initialize:
        initial_elements = make_associative_filtering_elements(model, observations, first_rank)
    else:
        initial_elements = observations

    def vectorized_operator(a, b):
        return tf.vectorized_map(filtering_operator, (a, b), fallback_to_while_loop=False)

    final_elements = tfp.math.scan_associative(
        vectorized_operator,
        initial_elements,
        max_num_levels=math.ceil(math.log2(max_parallel))
    )
    
    return final_elements

# ----------------------------
# Smoothing Element Functions
# ----------------------------

@tf.function(reduce_retracing=True)
def last_smoothing_element(m, P):
    """
    Returns the last smoothing element for a backward pass.

    Parameters
    ----------
    m : tf.Tensor
        Filtering mean at the last time step.
    P : tf.Tensor
        Filtering covariance at the last time step.

    Returns
    -------
    tuple of tf.Tensor
        (E, g, L) for the last element. Initialized to zeros and the final state.
    """
    return tf.zeros_like(P), m, P


# @partial(tf.function, experimental_relax_shapes=True)
@tf.function(reduce_retracing=True)
def generic_smoothing_element(model, m, P):
    """
    Computes the smoothing element for a single time step.

    Parameters
    ----------
    model : StateSpaceModel
        The state-space model.
    m : tf.Tensor
        Filtering mean at the given time step.
    P : tf.Tensor
        Filtering covariance at the given time step.

    Returns
    -------
    tuple of tf.Tensor
        (E, g, L) smoothing elements for this time step.
    """
    Pp = model.F @ mm(P, model.F, transpose_b=True) + model.Q
    chol = tf.linalg.cholesky(Pp)
    E = tf.transpose(tf.linalg.cholesky_solve(chol, model.F @ P))
    g = m - mv(E @ model.F, m)
    L = P - E @ mm(Pp, E, transpose_b=True)
    return E, g, L


# @partial(tf.function, experimental_relax_shapes=True)
@tf.function(reduce_retracing=True)
def make_associative_smoothing_elements(model, filtering_means, filtering_covariances, last_rank=False):
    """
    Prepare smoothing elements for all time steps for associative scan.

    Parameters
    ----------
    model : StateSpaceModel
        The state-space model.
    filtering_means : tf.Tensor
        Filtered state means, shape [T, dim].
    filtering_covariances : tf.Tensor
        Filtered state covariances, shape [T, dim, dim].
    last_rank : bool, optional
        Whether to treat the last element specially (default False).

    Returns
    -------
    tuple of tf.Tensor
        Smoothing elements (E, g, L) ready for associative scan.
    """
    if last_rank:
        last_elems = last_smoothing_element(filtering_means[-1], filtering_covariances[-1])
        generic_elems = tf.vectorized_map(
            lambda o: generic_smoothing_element(model, o[0], o[1]), 
            (filtering_means[:-1], filtering_covariances[:-1]),
            fallback_to_while_loop=False
        )
        return tuple(tf.concat([gen_es, tf.expand_dims(last_e, 0)], axis=0) 
                     for gen_es, last_e in zip(generic_elems, last_elems))
    else:
        generic_elems = tf.vectorized_map(
            lambda o: generic_smoothing_element(model, o[0], o[1]), 
            (filtering_means, filtering_covariances),
            fallback_to_while_loop=False
        )
        return tuple(generic_elems)


# ----------------------------
# Smoothing Operator
# ----------------------------

@tf.function(reduce_retracing=True)
def smoothing_operator(elems):
    """
    Combine two smoothing elements using the associative smoothing operator.

    Parameters
    ----------
    elems : tuple
        Tuple of two smoothing elements (elem1, elem2), each (E, g, L).

    Returns
    -------
    tuple of tf.Tensor
        The combined smoothing element (E, g, L).
    """
    elem1, elem2 = elems
    E1, g1, L1 = elem1
    E2, g2, L2 = elem2

    E = E2 @ E1
    g = mv(E2, g1) + g2
    L = E2 @ mm(L1, E2, transpose_b=True) + L2
    return E, g, L


class SmoothingOperator:
    """
    Callable class for the smoothing operator (E, g, L).
    """
    def __init__(self, dim):
        """
        Initialize smoothing operator with dimension `dim`, and the operator identity.
        """
        self.dim = dim
        self.identity = self.init_identity()

    @tf.function(reduce_retracing=True)
    def __call__(self, *args):
        """
        Apply smoothing operator to 1 or 2 elements.

        Parameters
        ----------
        args : tuple
            A tuple of two smoothing elements or two separate smoothing elements.

        Returns
        -------
        tuple
            Combined smoothing element (E, g, L).
        """
        if len(args) == 1:
            elems = args[0]
            return smoothing_operator(elems)
        elif len(args) == 2:
            a, b = args
            return smoothing_operator((a, b))
        else:
            raise ValueError(f"Expected 1 or 2 arguments, got {len(args)}")
    
    def init_identity(self):
        """
        Initialize identity element for smoothing (E=I, g=0, L=0).

        Returns
        -------
        list of tf.Tensor
            Identity smoothing element [E, g, L].
        """
        I = tf.eye(self.dim, dtype=tf.float64)
        Z = tf.zeros((self.dim,), dtype=tf.float64)
        ZM = tf.zeros((self.dim, self.dim), dtype=tf.float64)
        return I, Z, ZM  # [E, g, L]

@tf.function(reduce_retracing=True)
def pks(model, filtered_means, filtered_covariances, max_parallel=100000, last_rank=False):
    """
    Parallel Kalman Smoother (PKS) using associative scan.

    Parameters
    ----------
    model : StateSpaceModel
        The state-space model.
    filtered_means : tf.Tensor
        Filtered state means, shape [T, dim].
    filtered_covariances : tf.Tensor
        Filtered state covariances, shape [T, dim, dim].
    max_parallel : int, optional
        Maximum parallel operations (default 100,000).
    last_rank : bool, optional
        Whether to treat the last element specially (default False).

    Returns
    -------
    tuple of tf.Tensor
        Smoothed means, covariances, and L matrices: (ms, Ps, Ls)
    """
    initial_elements = make_associative_smoothing_elements(model, filtered_means, filtered_covariances, last_rank)
    reversed_elements = tuple(tf.reverse(elem, axis=[0]) for elem in initial_elements)

    def vectorized_operator(a, b):
        return tf.vectorized_map(smoothing_operator, (a, b), fallback_to_while_loop=False)

    final_elements = tfp.math.scan_associative(
        vectorized_operator, 
        reversed_elements, 
        max_num_levels=math.ceil(math.log2(max_parallel))
    )

    return (tf.reverse(final_elements[0], axis=[0]),  
            tf.reverse(final_elements[1], axis=[0]), 
            tf.reverse(final_elements[2], axis=[0]))

# ---------------------------------
# Filtering and Smoothing Interface
# ---------------------------------

@tf.function(reduce_retracing=True)
def pkfs(model, observations, max_parallel=100000, first_rank=True):
    """
    Parallel Kalman Filter + Smoother (PKFS) for a sequence of observations.

    Parameters
    ----------
    model : StateSpaceModel
        The state-space model.
    observations : tf.Tensor or np.ndarray
        Observations, shape [T, ydim].
    max_parallel : int, optional
        Maximum parallel operations for scan (default 100,000).
    first_rank : bool, optional
        Whether to treat the first rank specially (default True).

    Returns
    -------
    tuple of tf.Tensor
        Smoothed means, covariances, and L matrices: (ms, Ps, Ls)
    """
    return pks(
        model, 
        *pkf(model, observations, max_parallel, first_rank)[1:3],
        max_parallel,
        first_rank
    )

# ---------------------------------
# Batched Filtering and Smoothing
# ---------------------------------

@tf.function(reduce_retracing=True)
def pkf_batch(model, batch_obs, max_parallel=100000, last_r=None, batch_first_rank=False):
    """
    Parallel Kalman Filter for a batch of observations using associative scan.

    Parameters
    ----------
    model : StateSpaceModel
        The state-space model.
    batch_obs : tf.Tensor or np.ndarray
        Observations for the batch, shape [T_batch, ydim].
    max_parallel : int, optional
        Maximum parallel operations for the associative scan (default 100,000).
    last_r : tuple of tf.Tensor, optional
        Optional element from the previous batch to continue filtering.
    batch_first_rank : bool, optional
        Whether this is the first batch (default False).

    Returns
    -------
    tuple of tf.Tensor
        Final filtering elements (means, covariances, etc.) for this batch.
    """
    initial_elements = make_associative_filtering_elements(model, batch_obs, batch_first_rank)

    def vectorized_operator(a, b):
        return tf.vectorized_map(filtering_operator, (a, b), fallback_to_while_loop=False)

    if last_r is not None:
        initial_elements = tuple(tf.concat([prev_result, elem], axis=0)
                                 for elem, prev_result in zip(initial_elements, last_r))

    final_elements = tfp.math.scan_associative(
        vectorized_operator,
        initial_elements,
        max_num_levels=math.ceil(math.log2(max_parallel))
    )

    if last_r is not None:
        final_elements = tuple(elem[1:] for elem in final_elements)

    return final_elements


def batched_pkf(model, observations, max_parallel=100000, batch_size=2, first_rank=False):
    """
    Apply Kalman filtering in batches for large sequences of observations.

    Parameters
    ----------
    model : StateSpaceModel
        The state-space model.
    observations : tf.Tensor or np.ndarray
        Full sequence of observations, shape [T, ydim].
    max_parallel : int, optional
        Maximum parallel operations for associative scan (default 100,000).
    batch_size : int, optional
        Size of each batch for processing (default 2).
    first_rank : bool, optional
        Whether the first batch should be treated as first rank (default False).

    Returns
    -------
    tuple of tf.Tensor
        Concatenated filtering results (means, covariances, etc.) for all batches.
    """
    num_batches = math.ceil(len(observations) / batch_size)
    results = []

    for i in range(num_batches):
        batch_obs = observations[i * batch_size: (i + 1) * batch_size]

        if first_rank and i == 0:
            batch_first_rank = True
            last_r = None
        else:
            batch_first_rank = False
            last_r = tuple(elem[-1:, ...] for elem in results[-1])

        final_elements = pkf_batch(model, batch_obs, max_parallel, last_r, batch_first_rank)
        results.append(final_elements)

    results = tuple(tf.concat([res[i] for res in results], axis=0)
                        for i in range(len(results[0])))

    return results


@tf.function(reduce_retracing=True)
def pks_batch(model, batch_means, batch_covariances, max_parallel=1000000, last_r=None, batch_last_rank=False):
    """
    Parallel Kalman Smoother for a batch of filtered means and covariances using associative scan.

    Parameters
    ----------
    model : StateSpaceModel
        The state-space model.
    batch_means : tf.Tensor
        Filtered state means for the batch.
    batch_covariances : tf.Tensor
        Filtered state covariances for the batch.
    max_parallel : int, optional
        Maximum parallel operations for associative scan (default 100,000).
    last_r : tuple of tf.Tensor, optional
        Optional element from the previous batch to continue smoothing.
    batch_last_rank : bool, optional
        Whether this batch contains the last element for smoothing (default False).

    Returns
    -------
    tuple of tf.Tensor
        Final smoothing elements (E, g, L) for this batch.
    """
    initial_elements = make_associative_smoothing_elements(model, batch_means, batch_covariances, batch_last_rank)

    def vectorized_operator(a, b):
        return tf.vectorized_map(smoothing_operator, (a, b), fallback_to_while_loop=False)

    if last_r is not None:
        initial_elements = tuple(tf.concat([prev_result, elem], axis=0)
                                 for elem, prev_result in zip(initial_elements, last_r))

    final_elements = tfp.math.scan_associative(
        vectorized_operator,
        initial_elements,
        max_num_levels=math.ceil(math.log2(max_parallel))
    )

    if last_r is not None:
        final_elements = tuple(elem[1:] for elem in final_elements)

    return final_elements


def batched_pks(model, filtered_means, filtered_covariances, max_parallel=100000, batch_size=2, last_rank=False):
    """
    Apply smoothing in batches for large sequences of filtered means and covariances.

    Parameters
    ----------
    model : StateSpaceModel
        The state-space model.
    filtered_means : tf.Tensor
        Filtered state means, shape [T, dim].
    filtered_covariances : tf.Tensor
        Filtered state covariances, shape [T, dim, dim].
    max_parallel : int, optional
        Maximum parallel operations for associative scan (default 100,000).
    batch_size : int, optional
        Size of each batch for processing (default 2).
    last_rank : bool, optional
        Whether the first batch should be treated as last rank (default False).

    Returns
    -------
    tuple of tf.Tensor
        Concatenated smoothed means and covariances for all batches.
    """
    num_batches = math.ceil(filtered_means.shape[0] / batch_size)
    reversed_filtered_means = tf.reverse(filtered_means, axis=[0])
    reversed_filtered_covariances = tf.reverse(filtered_covariances, axis=[0])

    results = []

    for i in range(num_batches):
        batch_means = reversed_filtered_means[i * batch_size: (i + 1) * batch_size]
        batch_covariances = reversed_filtered_covariances[i * batch_size: (i + 1) * batch_size]

        if last_rank and i == 0:
            batch_last_rank = True
            last_r = None
        else:
            batch_last_rank = False
            last_r = tuple(elem[-1:, ...] for elem in results[-1])

        final_elements = pks_batch(model, batch_means, batch_covariances, max_parallel, last_r, batch_last_rank)
        results.append(final_elements)

    final_means = tf.reverse(tf.concat([res[0] for res in results], axis=0), axis=[0])
    final_covariances = tf.reverse(tf.concat([res[1] for res in results], axis=0), axis=[0])

    return final_means, final_covariances


def batched_pkfs(model, observations, max_parallel=100000, batch_size=2, first_rank=True):
    """
    Batched Parallel Kalman Filter + Smoother for large observation sequences.

    Parameters
    ----------
    model : StateSpaceModel
        The state-space model.
    observations : tf.Tensor or np.ndarray
        Observations, shape [T, ydim].
    max_parallel : int, optional
        Maximum parallel operations for associative scan (default 100,000).
    batch_size : int, optional
        Size of each batch (default 2).
    first_rank : bool, optional
        Whether the current process is the first MPI rank (default True).

    Returns
    -------
    tuple of tf.Tensor
        Smoothed means and covariances for the full sequence.
    """
    return batched_pks(
        model,
        *batched_pkf(model, observations, max_parallel, batch_size, first_rank)[1:3],
        max_parallel=max_parallel,
        batch_size=batch_size,
        last_rank=first_rank
    )
