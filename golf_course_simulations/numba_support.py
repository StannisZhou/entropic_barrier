import numba
from sympy.utilities.lambdify import lambdify
import numpy as np
from golf_course_simulations.utils import DEFAULT_THRESHOLD_MULTIPLIER

@numba.jit(nopython=True, nogil=True, cache=True)
def advance_flat_regions(current_location, centers, radiuses, time_step):
    boundary_radius = 1.
    scale = np.sqrt(time_step)
    n_dim = current_location.size
    n_three_spheres = centers.shape[0]
    boundary_radius_squared = 1
    radiuses_squared = radiuses**2
    distances_squared = np.zeros((n_three_spheres,))
    origin = np.zeros_like(current_location)
    threshold = DEFAULT_THRESHOLD_MULTIPLIER * scale
    temp_step_size = np.zeros(n_three_spheres + 1)
    while True:
        previous_location = current_location
        for jj in range(n_three_spheres):
            distances_squared[jj] = np.sum((current_location - centers[jj])**2)

        if not np.all(distances_squared > radiuses_squared):
            in_spheres = (distances_squared <= radiuses_squared)
            temp = np.nonzero(in_spheres)
            assert len(temp) == 1 and len(temp[0]) == 1, "Too many nonzeros"
            index = temp[0][0]
            break

        r_squared = np.sum(current_location**2)
        assert r_squared < boundary_radius_squared, 'Outside reflecting boundary.'
        temp_step_size[:n_three_spheres] = np.sqrt(distances_squared) - radiuses
        temp_step_size[-1] = boundary_radius - np.sqrt(r_squared)
        maximum_step_size = np.min(temp_step_size)

        if maximum_step_size > threshold:
            temp_current_location = uniform_on_sphere(current_location, maximum_step_size, 1)
            current_location = temp_current_location[0, :]
        else:
            random_component = scale * np.random.randn(n_dim)
            current_location = previous_location + random_component
            r_squared = np.sum(current_location**2)
            if r_squared > boundary_radius_squared:
                current_location = _reflecting_boundary(
                    origin, boundary_radius,
                    previous_location, current_location
                )

            if np.sum(current_location**2) > boundary_radius_squared:
                while True:
                    random_component = scale * np.random.randn(n_dim)
                    current_location = previous_location + random_component
                    if np.sum(current_location**2) <= boundary_radius_squared:
                        break

                    current_location = _reflecting_boundary(
                        origin, boundary_radius,
                        previous_location, current_location
                    )

                    if np.sum(current_location**2) <= boundary_radius_squared:
                        break

    return previous_location, current_location, index


@numba.jit(nopython=True, nogil=True, cache=True)
def _interpolate(point1, point2, center, target_radius):
    """_interpolate
    Find a point on the sphere with center as center and target_radius as radius.
    One of the two points (point1 and point2) should be inside the sphere, and
    the other one should be outside the shpere.
    Refer to the notes for the formulas used.

    Parameters
    ----------

    point1 : np array
        point1 is of shape (n,), where n is the dimension of the system
    point2 : np array
        point2 is of shape (n,), where n is the dimension of the system
    center : np array
        center is of shape (n,), where n is the dimension of the system
    target_radius : float
        target_radius is the radius of the sphere

    Returns
    -------

    point_on_sphere : np array
        point_on_sphere is of shape (n,), where n is the dimension of the system.
        We should have norm(point_on_sphere - center) == target_radius

    """
    a = np.sum((point2 - point1)**2)
    b = 2 * np.sum((point1 - center) * (point2 - point1))
    c = np.sum((point1 - center)**2) - target_radius**2
    roots = [
        (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a),
        (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
    ]
    if roots[0] > 0 and roots[0] < 1:
        assert roots[1] <= 0 or roots[1] >= 1, 'Found 2 valid solutions. Should have only 1.'
        lbda = roots[0]
    else:
        assert roots[1] > 0 and roots[1] < 1, "Didn't find a valid solution"
        lbda = roots[1]

    point_on_sphere = point1 + lbda * (point2 - point1)
    a = np.sum((point_on_sphere - center)**2)
    b = target_radius**2
    rtol=1.e-5
    atol=1.e-8
    assert np.abs(a - b) <= (atol + rtol * np.abs(b)), 'Given point is not on the sphere.'
    return point_on_sphere


@numba.jit(nopython=True, nogil=True, cache=True)
def _reflecting_boundary(center, radius, previous_location, current_location):
    radius_squared = radius**2
    assert np.sum((previous_location - center)**2) < radius_squared
    assert np.sum((current_location - center)**2) > radius_squared
    point_on_sphere = _interpolate(previous_location, current_location, center, radius)
    lbda = -np.sum((point_on_sphere - center) * (current_location - point_on_sphere)) / (2 * radius_squared)
    reflected_location = current_location + 4 * lbda * (point_on_sphere - center)
    return reflected_location


@numba.jit(nopython=True, cache=True)
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


def advance_within_concentric_spheres(current_location, three_sphere, boundary_radiuses, time_step):
    center = three_sphere.params['center']
    n_dim = center.size
    radiuses = three_sphere.params['radiuses']
    r1 = radiuses[1]
    assert boundary_radiuses[0] >= radiuses[0] and boundary_radiuses[1] <= radiuses[2]
    assert type(current_location) == np.ndarray
    assert current_location.shape == center.shape
    assert type(boundary_radiuses) == np.ndarray
    assert len(boundary_radiuses) == 2
    assert boundary_radiuses[0] < boundary_radiuses[1]
    distance = np.linalg.norm(current_location - center)
    assert distance > boundary_radiuses[0] and distance < boundary_radiuses[1]
    advance_within_concentric_spheres_numba = three_sphere.advance_within_concentric_spheres_numba
    previous_location, current_location, target_flag = advance_within_concentric_spheres_numba(
        current_location, center, r1, boundary_radiuses, time_step
    )

    return previous_location, current_location, target_flag
