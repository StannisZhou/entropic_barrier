import numpy as np
import logging
import pprint


DEFAULT_THRESHOLD_MULTIPLIER = 4
DEFAULT_RELATIVE_SCALE = 0.1


class ParamsProc(object):
    def __init__(self):
        self.keys = set()
        self.required_keys = set()
        self.optional_params = {}
        self.params_types = {}
        self.params_help = {}

    def add(self, key, param_type, help_info, default=None):
        if default is None:
            self.required_keys.add(key)
        else:
            assert type(default) is param_type
            self.optional_params[key] = default

        self.keys.add(key)
        self.params_types[key] = param_type
        self.params_help[key] = help_info

    def update(self, proc, excluded_params=[]):
        if not hasattr(self, 'keys'):
            self.keys = {}

        keys = proc.keys.difference(set(excluded_params))
        required_keys = proc.keys.difference(set(excluded_params))
        optional_params = {
            key: proc.optional_params[key] for key in proc.optional_params if key not in excluded_params
        }
        params_types = {
            key: proc.params_types[key] for key in proc.params_types if key not in excluded_params
        }
        params_help = {
            key: proc.params_help[key] for key in proc.params_help if key not in excluded_params
        }
        assert len(self.keys.intersection(keys)) == 0
        self.keys.update(keys)
        self.required_keys.update(required_keys)
        self.optional_params.update(optional_params)
        self.params_types.update(params_types)
        self.params_help.update(params_help)

    def get_empty_params(self):
        params = {}
        for key in self.required_keys:
            params[key] = None

        params.update(self.optional_params)
        params['keys_to_proc'] = []
        return params

    def process_params(self, params, params_proc=None, params_test=None, pp_params=True):
        assert self.required_keys.issubset(set(params.keys())), 'Required: {}, given: {}'.format(
            self.required_keys, params.keys()
        )
        for key in self.optional_params:
            if key not in params:
                params[key] = self.optional_params[key]

        assert set(params.keys()).issubset(self.keys), 'Given: {}, allowed: {}'.format(
            params.keys(), self.keys
        )
        for key in params:
            assert (
                type(params[key]) is self.params_types[key] or params[key] is None
            ), 'Key: {}, given type: {}, required type: {}'.format(
                key, type(params[key]), self.params_types[key]
            )

        if params_proc is not None:
            params_proc(params)

        if params_test is not None:
            params_test(params)

        if pp_params:
            logging.info(pprint.pformat(params))

        return params


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
        sample_norms[ii, 0] = np.sqrt(np.sum(samples[ii, :]**2))
    samples = radius * samples / sample_norms
    samples = samples + center.reshape((1, n))
    return samples


def _sample_uniform_initial_location(centers, radiuses, boundary_radius):
    """_sample_uniform_initial_location

    Parameters
    ----------

    centers : np array
        centers is an np array of shape (num_spheres, n_dim), and is the centers of all
        the three spheres.
    radiuses : np array
        radiuses is an np array of shape (num_spheres,), and is the radius of the target
        of each three sphere.
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
            distances = np.linalg.norm(initial_location.reshape((1, n)) - centers, ord=2, axis=1)
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
