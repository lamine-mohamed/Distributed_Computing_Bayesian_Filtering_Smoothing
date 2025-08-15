import numpy as np
from collections import namedtuple
import scipy as sc
from utils import make_car_tracking_model

# Named tuple for representing a linear Gaussian state-space model
StateSpaceModel = namedtuple(
    "StateSpaceModel", ["F", "H", "Q", "R", "m0", "P0", "xdim", "ydim"]
)

def get_data(model: StateSpaceModel, T: int, seed: int = 0):
    """
    Simulates data from a linear Gaussian state-space model.
    
    Parameters
    ----------
    model : StateSpaceModel
        The state-space model specifying system dynamics, observation model, 
        noise covariances, and initial state distribution.
    T : int
        Number of time steps to simulate.
    seed : int, optional
        Random seed for reproducibility (default: 0).
    
    Returns
    -------
    xs : np.ndarray, shape (T, xdim)
        Simulated True states.
    ys : np.ndarray, shape (T, ydim)
        Simulated noisy observations.
    """
    # Random number generator for reproducibility
    rng = np.random.RandomState(seed)
    normals = rng.randn(1 + T, model.xdim + model.ydim)
    
    # Allocate arrays for states and observations
    xs = np.empty((T, model.xdim))
    ys = np.empty((T, model.ydim))

    # Precompute Cholesky factors for process and measurement noise
    Q_chol = sc.linalg.cholesky(model.Q, lower=True)
    R_chol = sc.linalg.cholesky(model.R, lower=True)
    P0_chol = sc.linalg.cholesky(model.P0, lower=True)

    # Sample initial state
    x = model.m0 + P0_chol @ normals[0, :model.xdim]
    
    # Simulate system
    for i, norm in enumerate(normals[1:]):
        x = model.F @ x + Q_chol @ norm[:model.xdim]
        y = model.H @ x + R_chol @ norm[model.xdim:]
        xs[i] = x
        ys[i] = y
    
    return xs, ys


if __name__ == "__main__":
    # Create a constant-velocity car tracking model
    car_tracking_model = make_car_tracking_model(
        q=1., dt=0.1, r=0.5, 
        m0=np.array([0., 0., 1., -1.]), 
        P0=np.eye(4)
    )
    
    log10T = 8  # Number of time steps = 10^log10T
    T = 10 ** log10T
    
    # Simulate ground truth and observations
    true_xs, ys = get_data(car_tracking_model, T, seed=0)

    # Save results to disk
    np.savez("../data/car_tracking/observations9.npz", ys=ys)
    np.savez("../data/car_tracking/true_states9.npz", xs=true_xs)
    
    print("Data generated and saved successfully.")
