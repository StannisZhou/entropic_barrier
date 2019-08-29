from math import *

import numba
import numpy as np
import sympy
from sympy.utilities.lambdify import lambdastr

import golf_course.estimate.numba as nestimate


class Target(object):
    """
    Target
    A class that describes regions around important points in the energy function.
    Outside these these regions, the energy function would simply be a constant.
    """

    def __init__(self, center, radiuses, energy_type, energy_params):
        """__init__
        For 'well', there's only one parameter, 'depth'. We are going to use a quadratic energy function for this.
        For 'crater', there're two parameters, 'depth' and 'height'.  See the info in the notes about the energy
        function.  We will use a fourth-order polynomial. For 'random_well', there are three parameters, 'depth'
        for the well energy, 'locations' for the locations of the Gaussian bumps, and 'standard_deviations' for
        the standard_deviations of each multivariate Gaussian.  For each Gaussian bumps, we would start by assuming
        all the dimensions are independent of each other. But for each dimension, we can have a different
        standard_deviation params['locations'] is an np array of shape (num_loc, n), where num_loc is the number
        of Gaussian bumps we are going to put down, and n is the dimension of the system.
        params['standard_deviations'] is also an np array of shape (num_loc, n).  Here, each row holds the
        standard_deviations of all those dimensions. As a result, diag(params['standard_deviations'][i, :]**2) would
        be the covariance matrix for the ith Gaussian bump. For 'random_crater', there are four parameters. 'depth',
        'height', 'locations', and 'standard_deviations'. Refer to the above comments for the meaning of these.

        Parameters
        ----------
        center: np.array
            The location of the center for this Target. Length of the array is the dimension of the sysetm
        radiuses: np.array
            The radiuses of the three spheres involved. We should have radiuses[0]<radiuses[1]<radiuses[2],
        energy_type: str
            energy_type is the name of the type of energy we are going to use within the middle sphere.
            Allowed energy types include 'random_well' and 'random_crater'.  'random_well' and
            'random_crater' are two random energy functions. The way to define them is, we first
            randomly pick some locations, and put down some Gaussian bumps at those locations.
            We then multiply this energy function by either the the well energy or the crater energy,
            to get random_well and random crater.
        energy_params: dict
            The parameters for the spefic energy type that we are using
        Returns
        -------
        """
        assert len(radiuses) == 3
        assert (
            radiuses[0] < radiuses[1] and radiuses[1] < radiuses[2]
        ), 'Wrong radiuses.'
        assert energy_type in set(['random_well', 'random_crater', 'flat'])
        self.energy_type = energy_type
        self.center = center
        self.radiuses = radiuses
        self.energy_params = energy_params
        self.generate_force_field_function()

    def generate_force_field_function(self):
        if self.energy_type == 'flat':
            get_force_field = lambda x: list(np.zeros_like(x))
        else:
            expr_generation_func_dict = {
                'random_well': generate_random_well_sympy_expr,
                'random_crater': generate_random_crater_sympy_expr,
            }
            location, gradient_expr = expr_generation_func_dict[self.energy_type](
                self.center, self.radiuses, **self.energy_params
            )
            force_field_lambda_str = lambdastr(location, -gradient_expr)
            n_dim = len(location)
            old_argument = ','.join(['x{}'.format(ii) for ii in range(n_dim)])
            force_field_lambda_str = force_field_lambda_str.replace(old_argument, 'x')
            for ii in range(n_dim):
                force_field_lambda_str = force_field_lambda_str.replace(
                    'x{}'.format(ii), 'x[{}]'.format(ii)
                )

            get_force_field = eval(force_field_lambda_str)

        get_force_field = numba.jit(get_force_field)

        @numba.jit(nopython=True, cache=True)
        def advance_within_concentric_spheres_numba(
            current_location,
            center,
            r1,
            boundary_radiuses,
            time_step,
            reflecting_boundary_radius,
        ):
            origin = np.zeros_like(current_location)
            n_dim = center.size
            inner_boundary_squared = boundary_radiuses[0] ** 2
            outer_boundary_squared = boundary_radiuses[1] ** 2
            r1_squared = r1 ** 2
            scale = np.sqrt(time_step)
            previous_location = current_location
            target_flag = False
            while True:
                r_vector = current_location - center
                r_squared = np.sum(r_vector ** 2)
                if r_squared <= inner_boundary_squared:
                    target_flag = True
                    break
                elif r_squared >= outer_boundary_squared:
                    break

                if r_squared >= r1_squared:
                    force_field = np.zeros_like(current_location)
                else:
                    force_field = np.array(get_force_field(current_location))

                previous_location = current_location
                random_component = scale * np.random.randn(n_dim)
                current_location = (
                    previous_location + force_field * time_step + random_component
                )
                current_location = nestimate.simulate_reflecting_boundary(
                    origin,
                    reflecting_boundary_radius,
                    previous_location,
                    current_location,
                    scale,
                    time_step,
                    force_field,
                )

            return previous_location, current_location, target_flag

        self.advance_within_concentric_spheres_numba = (
            advance_within_concentric_spheres_numba
        )


def generate_random_well_sympy_expr(
    center,
    radiuses,
    depth=None,
    locations=None,
    standard_deviations=None,
    multiplier=None,
):
    """generate_random_well_sympy_expr
    Parameters
    ----------
    center :

    radiuses :

    depth : float
        The depth of the potential well
    locations : np.array
        The locations of all the Gaussian random bumps
    standard_deviations : np.array
        The standard deviations for the different Gaussian random bumps
    multiplier : float
        The multiplier used to balance the main and random parts of the energy function
    Returns
    -------
    """
    assert locations.shape[1] == center.size
    assert standard_deviations.shape == (locations.shape[0],)
    n_dim = center.size
    n_bumps = locations.shape[0]
    location = sympy.Array(sympy.symbols('x:{}'.format(n_dim)), (n_dim,))
    center = sympy.Array(center, center.shape)
    r_squared = sympy_array_squared_norm(location - center)
    well_expr = (
        -(depth / radiuses[1] ** 4)
        * (r_squared ** 2 - 2 * radiuses[1] ** 2 * r_squared)
        - depth
    )
    mollifier_expr = sympy.functions.exp(
        -radiuses[1] / (radiuses[1] - sympy_array_squared_norm(location - center) ** 10)
    ) / np.exp(-1)
    random_components = [
        sympy.functions.exp(
            -sympy_array_squared_norm(location - sympy.Array(locations[ii], (n_dim,)))
            / (2 * standard_deviations[ii] ** 2)
        )
        for ii in range(n_bumps)
    ]
    random_expr = 0
    for ii in range(n_bumps):
        random_expr += random_components[ii]

    sympy_expr = well_expr + multiplier * mollifier_expr * random_expr
    gradient_expr = sympy.derive_by_array(sympy_expr, location)
    return location, gradient_expr


def generate_random_crater_sympy_expr(
    center,
    radiuses,
    depth=None,
    height=None,
    locations=None,
    standard_deviations=None,
    multiplier=None,
):
    """generate_random_crater_sympy_expr
    Parameters
    ----------
    center :

    radiuses :

    depth : float
        The depth of the crater
    height : float
        The height of the crater
    locations : np.array
        The locations of all the Gaussian random bumps
    standard_deviations : np.array
        The standard deviations for the different Gaussian random bumps
    multiplier : float
        The multiplier used to balance the main and the random parts of the energy function
    Returns
    -------
    """
    assert locations.shape[1] == center.size
    assert standard_deviations.shape == (locations.shape[0],)
    n_dim = center.size
    n_bumps = locations.shape[0]
    location = sympy.Array(sympy.symbols('x:{}'.format(n_dim)), (n_dim,))
    center = sympy.Array(center, center.shape)
    r_squared = sympy_array_squared_norm(location - center)
    C = (
        3
        * radiuses[1] ** 2
        * sympy.cbrt(depth * height * (depth + sympy.sqrt(depth * (depth + height))))
    )
    Delta0 = -9 * depth * height * radiuses[1] ** 4
    b_squared = -(1 / (3 * depth)) * (-3 * depth * radiuses[1] ** 2 + C + Delta0 / C)
    a = depth / (3 * b_squared * radiuses[1] ** 4 - radiuses[1] ** 6)
    crater_expr = (
        a
        * (
            2 * r_squared ** 3
            - 3 * (b_squared + radiuses[1] ** 2) * r_squared ** 2
            + 6 * b_squared * radiuses[1] ** 2 * r_squared
        )
        - depth
    )
    mollifier_expr = sympy.functions.exp(
        -radiuses[1] / (radiuses[1] - sympy_array_squared_norm(location - center) ** 10)
    ) / np.exp(-1)
    random_components = [
        sympy.functions.exp(
            -sympy_array_squared_norm(location - sympy.Array(locations[ii], (n_dim,)))
            / (2 * standard_deviations[ii] ** 2)
        )
        for ii in range(n_bumps)
    ]
    random_expr = 0
    for ii in range(n_bumps):
        random_expr += random_components[ii]

    sympy_expr = crater_expr + multiplier * mollifier_expr * random_expr
    gradient_expr = sympy.derive_by_array(sympy_expr, location)
    return location, gradient_expr


def sympy_array_squared_norm(sympy_array):
    return sympy.tensor.array.tensorcontraction(
        sympy_array.applyfunc(lambda x: x ** 2), (0,)
    )
