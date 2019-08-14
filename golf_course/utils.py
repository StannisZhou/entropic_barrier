import numpy as np

DEFAULT_THRESHOLD_MULTIPLIER = 4
DEFAULT_RELATIVE_SCALE = 0.1


def uniform_on_sphere(center, radius, num_samples=1):
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

    Returns
    -------

    samples : np array
        samples is an np array of shape (num_samples, center.size).
        Each row is a sample

    """
    n = center.size
    samples = np.random.randn(num_samples, n)
    sample_norms = np.zeros((num_samples, 1))
    for ii in range(num_samples):
        sample_norms[ii, 0] = np.sqrt(np.sum(samples[ii, :] ** 2))
    samples = radius * samples / sample_norms
    samples = samples + center.reshape((1, n))
    return samples


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
