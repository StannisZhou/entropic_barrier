import pickle

import numpy as np

DEFAULT_THRESHOLD_MULTIPLIER = 4
DEFAULT_RELATIVE_SCALE = 0.1


def uniform_on_sphere(center, radius, num_samples=1, reflecting_boundary_radius=np.inf):
    """uniform_on_sphere
    Uniform distribution on a sphere

    Parameters
    ----------

    center : np array
        center is the center of the sphere
    radius : float
        radius is the radius of the sphere
    num_samples : int
        num_samples is the number of samples we are going to get
    reflecting_boundary_radius : float
        The radius of the reflecting boundary. Gives us a further constraint that all
        the samples need to be within the reflecting boundary

    Returns
    -------

    valid_samples : np array
        samples is an np array of shape (num_samples, center.size).
        Each row is a sample

    """
    n = center.size
    n_valid = 0
    valid_samples = []
    while n_valid < num_samples:
        samples = np.random.randn(num_samples, n)
        sample_norms = np.linalg.norm(samples, axis=1, keepdims=True)
        samples = radius * samples / sample_norms
        samples = samples + center.reshape((1, n))
        samples = samples[np.linalg.norm(samples, axis=1) < reflecting_boundary_radius]
        valid_samples.append(samples)
        n_valid += len(samples)

    valid_samples = np.concatenate(valid_samples)
    assert np.all(np.linalg.norm(valid_samples, axis=1) < reflecting_boundary_radius)
    return valid_samples[:num_samples]


def sample_uniform_initial_location(centers, radiuses, boundary_radius):
    """sample_uniform_initial_location

    Parameters
    ----------

    centers : np array
        centers is an np array of shape (num_spheres, n_dim), and is the centers of all
        the targets.
    radiuses : np array
        radiuses is an np array of shape (num_spheres,), and is the radius of the targets.
    boundary_radius : float
        boundary_radius is the boundary radius of the reflecting boundary.

    Returns
    -------

    initial_location : np array
        initial_location is an np array of shape (n,), where n is the dimension of the
        system. initial_location is the location we sampled.

    """
    n = centers.shape[1]
    while True:
        initial_location = 2 * boundary_radius * np.random.rand(n) - boundary_radius
        if np.linalg.norm(initial_location, ord=2) < boundary_radius:
            distances = np.linalg.norm(
                initial_location.reshape((1, n)) - centers, ord=2, axis=1
            )
            if all(distances > radiuses):
                break

    return initial_location


def sample_random_locations(center, radiuses, n_samples):
    assert radiuses[0] < radiuses[1]
    n_dim = center.size
    locations = np.zeros((n_samples, n_dim))
    for ii in range(n_samples):
        random_radius = (radiuses[1] - radiuses[0]) * np.random.rand(1) + radiuses[0]
        locations[ii] = uniform_on_sphere(center, random_radius)

    return locations


def load_model_params(model_params_fname):
    with open(model_params_fname, 'rb') as f:
        model_params = pickle.load(f)

    time_step = model_params['time_step']
    target_param_list = model_params['target_param_list']
    return time_step, target_param_list
