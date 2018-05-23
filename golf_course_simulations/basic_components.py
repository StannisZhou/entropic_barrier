import numpy as np
import golf_course_simulations.numba_support as ns
from golf_course_simulations.utils import ParamsProc, DEFAULT_RELATIVE_SCALE, uniform_on_sphere
import sympy
from sympy.utilities.lambdify import lambdastr
from tqdm import tqdm
from math import *
from scipy.cluster.vq import kmeans2
import numba
import multiprocessing as mp
import functools
import timeit


class ThreeSphere(object):
    """
    ThreeSphere
    A class that describes regions around important points in the energy function.
    Outside these three spheres, the energy function would simply be a constant.
    """
    @staticmethod
    def get_proc():
        """
        For 'well', there's only one parameter, 'depth'. We are going to use a
        quadratic energy function for this.
        For 'crater', there're two parameters, 'depth' and 'height'.
        See the info in the notes about the energy function.
        We will use a fourth-order polynomial.
        For 'random_well', there are three parameters, 'depth' for the well energy,
        'locations' for the locations of the Gaussian bumps, and 'standard_deviations'
        for the standard_deviations of each multivariate Gaussian.
        For each Gaussian bumps, we would start by assuming all the dimensions
        are independent of each other. But for each dimension, we can have a
        different standard_deviation
        params['locations'] is an np array of shape (num_loc, n), where num_loc is
        the number of Gaussian bumps we are going to put down, and n is the
        dimension of the system.
        params['standard_deviations'] is also an np array of shape (num_loc, n).
        Here, each row holds the standard_deviations of all those dimensions. As a result,
        diag(params['standard_deviations'][i, :]**2) would be the covariance matrix
        for the ith Gaussian bump.
        For 'random_crater', there are four parameters. 'depth', 'height', 'locations',
        and 'standard_deviations'. Refer to the above comments for the meaning of these.
        For both 'random_well' and 'random_crater', there will be an optional
        'relative_scale' parameter, which helps us put the two parts of the energy
        on equal footing.
        """
        proc = ParamsProc()
        proc.add(
            'time_step', float,
            'The time step we are going to use for the simulation',
            1e-5
        )
        proc.add(
            'center', np.ndarray,
            'The location of the center for this ThreeSphere. Length of the array is the dimension of the sysetm'
        )
        proc.add(
            'radiuses', np.ndarray,
            'The radiuses of the three spheres involved. We should have radiuses[0]<radiuses[1]<radiuses[2]'
        )
        proc.add(
            'relative_scale', float,
            'The relative scale with which we are going to balance the main and the random part',
            DEFAULT_RELATIVE_SCALE
        )
        proc.add(
            'energy_type', str,
            '''
            energy_type is the name of the type of energy we are going to use within the middle sphere. Allowed
            energy types include 'random_well' and 'random_crater'.
            'random_well' and 'random_crater' are two random energy functions. The way
            to define them is, we first randomly pick some locations, and put down
            some Gaussian bumps at those locations. We then multiply this energy function
            by either the the well energy or the crater energy, to get random_well and random crater.
            '''
        )
        proc.add(
            'energy_params', dict,
            'The parameters for the spefic energy type that we are using'
        )
        proc.add(
            'inner', int,
            'The number of intermediate layers we are going to use for calculating the inner rate'
        )
        proc.add(
            'outer', int,
            'The number of intermediate layers we are going to use for calculating the outer rate'
        )
        proc.add(
            'num_points', int,
            '''
            The number of points we are going to have on each layer.
            We are going to form clusters based on these points.
            '''
        )
        proc.add(
            'num_clusters', int,
            'The number of clusters we are going to have on each layer'
        )
        proc.add(
            'num_trials', int,
            'The number of trials we are going to run for each bin in order to decide the transition probabilities'
        )
        proc.add(
            'use_parallel', bool,
            'Whether we are going to make the code parallel or not',
            True
        )
        proc.add(
            'n_split', int,
            'The number of splits we are going to use for making things parallel',
            4
        )
        return proc

    def __init__(self, params):
        proc = self.get_proc()
        def params_proc(params):
            if params['energy_type'] != 'flat':
                standard_deviations = params['energy_params']['standard_deviations']
                relative_scale = params['relative_scale']
                if params['energy_type'] == 'random_well':
                    energy_range = params['energy_params']['depth']
                elif params['energy_type'] == 'random_crater':
                    energy_range = params['energy_params']['depth'] + params['energy_params']['height']
                else:
                    assert False, 'Wrong energy type.'

                params['energy_params']['multiplier'] = float((params['relative_scale'] * energy_range))

        def params_test(params):
            assert len(params['radiuses']) == 3
            assert params['radiuses'][0] < params['radiuses'][1] and \
                params['radiuses'][1] < params['radiuses'][2], 'Wrong radiuses.'
            assert params['energy_type'] in set(['random_well', 'random_crater', 'flat'])

        self.params = proc.process_params(params, params_proc=params_proc, params_test=params_test)

    def generate_force_field_function(self):
        if self.params['energy_type'] == 'flat':
            get_force_field = (lambda x: list(np.zeros_like(x)))
        else:
            expr_generation_func_dict = {
                'random_well': generate_random_well_sympy_expr,
                'random_crater': generate_random_crater_sympy_expr
            }
            location, gradient_expr = expr_generation_func_dict[self.params['energy_type']](
                self.params['center'], self.params['radiuses'], self.params['energy_params'])
            force_field_lambda_str = lambdastr(location, -gradient_expr)
            n_dim = len(location)
            old_argument = ','.join(['x{}'.format(ii) for ii in range(n_dim)])
            force_field_lambda_str = force_field_lambda_str.replace(old_argument, 'x')
            for ii in range(n_dim):
                force_field_lambda_str = force_field_lambda_str.replace('x{}'.format(ii), 'x[{}]'.format(ii))

            get_force_field = eval(force_field_lambda_str)

        get_force_field = numba.jit(get_force_field)
        @numba.jit(nopython=True, cache=True)
        def advance_within_concentric_spheres_numba(current_location, center, r1, boundary_radiuses, time_step):
            n_dim = center.size
            inner_boundary_squared = boundary_radiuses[0]**2
            outer_boundary_squared = boundary_radiuses[1]**2
            r1_squared = r1**2
            scale = np.sqrt(time_step)
            previous_location = current_location
            target_flag = False
            while True:
                r_vector = current_location - center
                r_squared = np.sum(r_vector**2)
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

                current_location = previous_location +\
                    force_field * time_step + random_component

            return previous_location, current_location, target_flag

        self.advance_within_concentric_spheres_numba = advance_within_concentric_spheres_numba

    def estimate_capacity(self):
        forward_probabilities, backward_probabilities, cluster_labels = self._get_data_driven_binning_transition_probabilities()
        print('Transition probabilities calculation done.')
        probability = self._get_data_driven_binning_hitting_probability(
            forward_probabilities, backward_probabilities
        )
        middle_index = self.params['outer'] + 1
        cluster_labels = cluster_labels[middle_index]
        n_points_in_clusters = np.array([np.sum(cluster_labels == ii) for ii in range(self.params['num_clusters'])])
        n_dim = self.params['center'].size
        dA = self.params['radiuses'][1]
        rAtilde = self.params['radiuses'][2]
        capacity = (n_dim - 2) / (dA**(2 - n_dim) - rAtilde**(2 - n_dim)) * np.sum(
            n_points_in_clusters * probability
        ) / self.params['num_points']
        return capacity

    def _get_data_driven_binning_transition_probabilities(self):
        forward_probabilities = []
        backward_probabilities = []
        cluster_centers, cluster_labels, propagated_points, statistics_from_propagation = self._propagate_and_cluster()
        forward_probabilities, backward_probabilities = self._additional_simulations_for_transition_probabilities(
                cluster_centers, cluster_labels, propagated_points, statistics_from_propagation
        )
        return forward_probabilities, backward_probabilities, cluster_labels

    def _propagate_and_cluster(self):
        num_points = self.params['num_points']
        center = self.params['center']
        radiuses = self.params['radiuses']
        initial_locations = uniform_on_sphere(center, radiuses[1], num_points)
        inner = self.params['inner']
        outer = self.params['outer']
        num_surfaces = inner + outer + 3
        middle_index = outer + 1
        surfaces = self._get_surfaces()
        assert len(surfaces) == num_surfaces, 'The generated surfaces are not right.'

        # Propagate the points and gather information
        propagated_points = [[] for _ in range(num_surfaces)]
        propagated_points[middle_index] = initial_locations
        propagated_information = []
        extra_information = []
        print('Doing propagation.')
        # Do the initial propagation from the middle sphere
        self._propagate_and_get_info(
            surfaces, propagated_points, propagated_information, extra_information, middle_index
        )
        # Do the forward propagation, from the middle sphere to the inner sphere
        for index in range(middle_index + 1, num_surfaces - 1):
            self._propagate_and_get_info(
                surfaces, propagated_points, propagated_information, extra_information, index
            )
        # Do the backward propagation, from the middle sphere to the outer sphere
        for index in range(middle_index - 1, 0, -1):
            self._propagate_and_get_info(
                surfaces, propagated_points, propagated_information, extra_information, index
            )
        # Do the clustering
        num_clusters = self.params['num_clusters']
        cluster_centers = [[] for _ in range(num_surfaces)]
        cluster_labels = [[] for _ in range(num_surfaces)]
        print('Doing clustering.')
        for ii in tqdm(range(num_surfaces)):
            cluster_centers[ii], cluster_labels[ii] = kmeans2(
                propagated_points[ii], num_clusters, minit='points', missing='raise'
            )

        # Get the statistics
        print('Getting statistics.')
        statistics_from_propagation = self._collect_statistics(
            cluster_centers, cluster_labels, propagated_information, extra_information
        )

        return cluster_centers, cluster_labels, propagated_points,statistics_from_propagation

    def _get_surfaces(self):
        radiuses = self.params['radiuses']
        inner = self.params['inner']
        outer = self.params['outer']
        inner_surfaces = np.linspace(radiuses[1], radiuses[0], inner + 2)
        outer_surfaces = np.linspace(radiuses[2], radiuses[1], outer + 2)
        surfaces = np.concatenate((outer_surfaces, inner_surfaces[1:]))
        return surfaces

    def _propagate_and_get_info(self, surfaces, propagated_points, propagated_information, extra_information, index):
        num_points = self.params['num_points']
        assert propagated_points[index].shape[0] == num_points, 'Number of points not right.'
        time_step = self.params['time_step']
        boundary_radiuses = np.array([surfaces[index + 1], surfaces[index - 1]])
        with tqdm() as pbar:
            batch_size = 500
            while True:
                flag = False
                random_indices = np.random.randint(0, num_points, size=(batch_size,))
                initial_locations = propagated_points[index][random_indices]
                ii = 0
                indicator_list = []
                final_point_list = []
                for initial_location in initial_locations:
                    previous_location, current_location, target_flag = ns.advance_within_concentric_spheres(
                        initial_location, self, boundary_radiuses, time_step
                    )
                    if target_flag:
                        indicator = 1
                    else:
                        indicator = -1

                    final_point = ns._interpolate(
                        previous_location, current_location, self.params['center'], surfaces[index + indicator]
                    )
                    if len(propagated_points[index + indicator]) == num_points:
                        extra_temp = np.concatenate(
                            (np.array([index, random_indices[ii], index + indicator]),
                            final_point)
                        )
                        extra_information.append(extra_temp)
                    else:
                        propagated_points[index + indicator].append(final_point)
                        index_temp = len(propagated_points[index + indicator]) - 1
                        propagated_information.append(
                            np.array([index, random_indices[ii],
                                    index + indicator, index_temp], dtype=int)
                        )

                    pbar.update()
                    ii += 1
                    if len(propagated_points[index + 1]) == num_points\
                            and len(propagated_points[index - 1]) == num_points:
                        propagated_points[index + 1] = np.array(propagated_points[index + 1])
                        propagated_points[index - 1] = np.array(propagated_points[index - 1])
                        flag = True
                        break

                if flag:
                    break

    def _collect_statistics(self, cluster_centers, cluster_labels, propagated_information, extra_information):
        num_surfaces = self.params['inner'] + self.params['outer'] + 3
        num_clusters = self.params['num_clusters']
        statistics_from_propagation = [[[] for _ in range(num_clusters)] for _ in range(num_surfaces)]
        self._process_propagated_info(cluster_labels, statistics_from_propagation, propagated_information)
        self._process_extra_info(cluster_centers, cluster_labels, statistics_from_propagation, extra_information)
        return statistics_from_propagation

    def _process_extra_info(self, cluster_centers, cluster_labels, statistics_from_propagation, extra_information):
        for info in extra_information:
            centers = cluster_centers[int(info[2])]
            point = info[3:]
            info_temp = info[:3].astype(int)
            index = self._assign_clusters(point, centers)
            statistics_from_propagation[info_temp[0]][cluster_labels[info_temp[0]][info_temp[1]]].append(
                (info_temp[2], index)
            )

    def _assign_clusters(self, point, centers):
        distances = np.linalg.norm(point - centers, ord=2, axis=1)
        index = np.argmin(distances)
        return index

    def _process_propagated_info(self, cluster_labels, statistics_from_propagation, propagated_information):
        for info in propagated_information:
            statistics_from_propagation[info[0]][cluster_labels[info[0]][info[1]]].append(
                (info[2], cluster_labels[info[2]][info[3]])
            )

    def _additional_simulations_for_transition_probabilities(
        self, cluster_centers, cluster_labels, propagated_points, statistics_from_propagation
    ):
        surfaces = self._get_surfaces()
        num_surfaces = len(surfaces)
        num_clusters = self.params['num_clusters']
        if self.params['use_parallel']:
            manager = mp.Manager()
            statistics_from_propagation = [
                [manager.list(level3) for level3 in level2] for level2 in statistics_from_propagation
            ]

        print('Doing additional simulations.')
        # Do more simulations and update statistics_from_propagation
        for ii in range(1, num_surfaces - 1):
            for jj in range(num_clusters):
                print('Doing simulations for surface {}, cluster {}.'.format(ii, jj))
                self._do_additional_simulations(
                    ii, jj, cluster_centers, cluster_labels, propagated_points, statistics_from_propagation
                )

        if self.params['use_parallel']:
            for ii in range(len(statistics_from_propagation)):
                statistics_from_propagation[ii] = [list(level3) for level3 in statistics_from_propagation[ii]]

        # Use statistics_from_propagation to calculate forward and backward probabilities
        forward_probabilities, backward_probabilities = self._process_statistics_from_propagation(
            statistics_from_propagation
        )
        return forward_probabilities, backward_probabilities

    def _do_additional_simulations(
        self, surface_index, cluster_index, cluster_centers, cluster_labels,
        propagated_points, statistics_from_propagation
    ):
        num_trials = self.params['num_trials']
        time_step = self.params['time_step']
        surfaces = self._get_surfaces()
        cluster_points_indices = np.flatnonzero(cluster_labels[surface_index] == cluster_index)
        cluster_size = cluster_points_indices.size
        random_indices = np.random.randint(0, cluster_size, size=(num_trials,))
        initial_locations = propagated_points[surface_index][cluster_points_indices[random_indices]]
        boundary_radiuses = np.array([surfaces[surface_index + 1], surfaces[surface_index - 1]])
        if self.params['use_parallel']:
            n_locations = initial_locations.shape[0]
            def worker(indices, q):
                for index in indices:
                    initial_location = initial_locations[index]
                    previous_location, current_location, target_flag = ns.advance_within_concentric_spheres(
                        initial_location, self, boundary_radiuses, time_step
                    )
                    if target_flag:
                        indicator = 1
                    else:
                        indicator = -1

                    final_point = ns._interpolate(
                        previous_location, current_location, self.params['center'], surfaces[surface_index + indicator]
                    )
                    centers = cluster_centers[surface_index + indicator]
                    index = self._assign_clusters(final_point, centers)
                    statistics_from_propagation[surface_index][cluster_index].append(
                        (surface_index + indicator, index)
                    )
                    q.put(1)

            process_list = []
            q = mp.Queue()
            def listener(q):
                with tqdm(total=n_locations) as pbar:
                    for item in iter(q.get, None):
                        pbar.update()

            listener_process = mp.Process(target=listener, args=(q,))
            listener_process.start()
            for indices in kfold_split(n_locations, self.params['n_split']):
                process = mp.Process(
                    target=worker, args=(indices, q)
                )
                process_list.append(process)
                process.start()

            for process in process_list:
                process.join()

            q.put(None)
            listener_process.join()
        else:
            for initial_location in tqdm(initial_locations):
                previous_location, current_location, target_flag = ns.advance_within_concentric_spheres(
                    initial_location, self, boundary_radiuses, time_step
                )
                if target_flag:
                    indicator = 1
                else:
                    indicator = -1

                final_point = ns._interpolate(
                    previous_location, current_location, self.params['center'], surfaces[surface_index + indicator]
                )
                centers = cluster_centers[surface_index + indicator]
                index = self._assign_clusters(final_point, centers)
                statistics_from_propagation[surface_index][cluster_index].append(
                    (surface_index + indicator, index)
                )

    def _process_statistics_from_propagation(self, statistics_from_propagation):
        num_surfaces = len(statistics_from_propagation)
        num_clusters = self.params['num_clusters']
        forward_probabilities = [np.zeros((num_clusters, num_clusters), dtype=float)
                                 for _ in range(num_surfaces - 1)]
        backward_probabilities = [np.zeros((num_clusters, num_clusters), dtype=float)
                                  for _ in range(num_surfaces - 1)]
        for ii in range(1, num_surfaces - 1):
            for jj in range(num_clusters):
                statistics_temp = np.array(statistics_from_propagation[ii][jj])
                forward_transitions = statistics_temp[statistics_temp[:, 0] == ii + 1, 1]
                backward_transitions = statistics_temp[statistics_temp[:, 0] == ii - 1, 1]
                forward_frequencies = np.bincount(forward_transitions, minlength=num_clusters)
                backward_frequencies = np.bincount(backward_transitions, minlength=num_clusters)
                assert forward_frequencies.size == num_clusters and\
                    backward_frequencies.size == num_clusters,\
                    "Frequencies not right. ff : {}, bf: {}, nc: {}, ft: {}, bt: {}".format(
                        forward_frequencies.size, backward_frequencies.size, num_clusters,
                        forward_transitions, backward_transitions
                    )
                total_transitions = float(np.sum(forward_frequencies) + np.sum(backward_frequencies))
                assert total_transitions == statistics_temp.shape[0],\
                    '#transitions: {}, forward_frequencies: {}, backward_frequencies: {}, sum: {}'.format(
                        total_transitions, forward_frequencies, backward_frequencies,
                    )
                forward_frequencies = forward_frequencies.astype(float)
                backward_frequencies = backward_frequencies.astype(float)
                forward_probabilities[ii][jj, :] = forward_frequencies / total_transitions
                backward_probabilities[ii][jj, :] = backward_frequencies / total_transitions

        return forward_probabilities, backward_probabilities

    def _get_data_driven_binning_hitting_probability(self, forward_probabilities, backward_probabilities):
        inner = self.params['inner']
        outer = self.params['outer']
        num_surfaces = inner + outer + 3
        middle_index = outer + 1
        num_clusters = self.params['num_clusters']
        un = np.ones((num_clusters,))
        u0 = np.zeros((num_clusters,))
        Q_matrices = [[] for _ in range(num_surfaces)]
        for jj in range(1, num_surfaces - 1):
            try:
                inverse_forward = np.linalg.inv(forward_probabilities[jj])
            except:
                epsilon = 1e-3 * np.min(forward_probabilities[jj][forward_probabilities[jj] > 0])
                temp_sum = np.sum(forward_probabilities[jj], axis=1, keepdims=True)
                forward_probabilities[jj] = forward_probabilities[jj] +\
                    epsilon * np.eye(num_clusters)
                forward_probabilities[jj] = temp_sum * forward_probabilities[jj] /\
                    np.sum(forward_probabilities[jj], axis=1, keepdims=True)
                assert np.alltrue(np.sum(forward_probabilities[jj], axis=1) == temp_sum)
                inverse_forward = np.linalg.inv(forward_probabilities[jj])

            matrix_A = inverse_forward
            matrix_B = - np.dot(inverse_forward, backward_probabilities[jj])
            matrix_C = np.eye(num_clusters)
            matrix_D = np.zeros((num_clusters, num_clusters))
            matrix_temp = np.concatenate(
                (
                    np.concatenate((matrix_A, matrix_B), axis=1),
                    np.concatenate((matrix_C, matrix_D), axis=1)
                ), axis=0
            )
            Q_matrices[jj] = matrix_temp

        Q_product = np.eye(2 * num_clusters)
        for jj in range(1, num_surfaces - 1):
            Q_product = np.dot(Q_matrices[jj], Q_product)
            if jj == middle_index - 1:
                Q_middle = Q_product

        u1 = np.linalg.solve(
            Q_product[:num_clusters, :num_clusters], un - np.dot(Q_product[:num_clusters, num_clusters:], u0)
        )

        temp = np.dot(Q_middle, np.concatenate((u1, u0)))
        probability = temp[:num_clusters]
        return probability


class ToyModel(object):
    """
    ToyModel
    This class defines the toy model we are going to use.
    For simplicity, in our program, we will simply use 0 for the constant region
    outside all the three-spheres.
    """
    @staticmethod
    def get_proc():
        proc = ParamsProc()
        proc.add(
            'time_step', float,
            'The time step we are going to use for the simulation',
            1e-5
        )
        proc.add(
            'three_sphere_param_list', list,
            'A list of dict, each of which contains the parameters for a particular three sphere'
        )
        return proc

    def __init__(self, params):
        proc = self.get_proc()
        def params_proc(params):
            for three_sphere_param in params['three_sphere_param_list']:
                if 'time_step' not in three_sphere_param:
                    three_sphere_param['time_step'] = params['time_step']

        def params_test(params):
            assert _check_compatibility(params['three_sphere_param_list'], 1)

        self.params = proc.process_params(params, params_proc=params_proc, params_test=params_test)
        self.three_sphere_list = [
            ThreeSphere(params) for params in self.params['three_sphere_param_list']
        ]
        print('Generating force field function')
        for three_sphere in tqdm(self.three_sphere_list):
            three_sphere.generate_force_field_function()

    def do_naive_simulation(self, current_location):
        radiuses = np.array(
            [three_sphere.params['radiuses'][1] for three_sphere in self.three_sphere_list]
        )
        centers = np.array(
            [three_sphere.params['center'] for three_sphere in self.three_sphere_list]
        )
        distances = np.linalg.norm(current_location.reshape((1, -1)) - centers, ord=2, axis=1)
        assert all(distances > radiuses)
        while True:
            previous_location, current_location, index = ns.advance_flat_regions(
                current_location, centers, radiuses, self.params['time_step']
            )
            three_sphere = self.three_sphere_list[index]
            boundary_radiuses = np.array(
                [three_sphere.params['radiuses'][0], three_sphere.params['radiuses'][1]]
            )
            previous_location, current_location, target_flag = ns.advance_within_concentric_spheres(
                current_location, three_sphere, boundary_radiuses, self.params['time_step'],
            )
            if target_flag:
                break

        return index

    def estimate_hitting_prob(self):
        n_targets = len(self.three_sphere_list)
        hitting_prob = np.zeros(n_targets)
        for ii in range(n_targets):
            three_sphere = self.three_sphere_list[ii]
            hitting_prob[ii] = three_sphere.estimate_capacity()

        hitting_prob = hitting_prob / np.sum(hitting_prob)
        return hitting_prob


def _check_compatibility(three_sphere_param_list, boundary_radius):
    n_spheres = len(three_sphere_param_list)
    for three_sphere_param in three_sphere_param_list:
        center = three_sphere_param['center']
        radius = three_sphere_param['radiuses'][2]
        if np.linalg.norm(center) + radius > boundary_radius:
            return False

    for ii in range(n_spheres - 1):
        for jj in range(ii + 1, n_spheres):
            center1 = three_sphere_param_list[ii]['center']
            radius1 = three_sphere_param_list[ii]['radiuses'][2]
            center2 = three_sphere_param_list[jj]['center']
            radius2 = three_sphere_param_list[jj]['radiuses'][2]
            if np.linalg.norm(center1 - center2) < radius1 + radius2:
                return False

    return True


def generate_random_well_sympy_expr(center, radiuses, energy_params):
    proc = ParamsProc()
    proc.add(
        'depth', float,
        'The depth of the potential well'
    )
    proc.add(
        'locations', np.ndarray,
        'The locations of all the Gaussian random bumps'
    )
    proc.add(
        'standard_deviations', np.ndarray,
        'The standard deviations for the different Gaussian random bumps'
    )
    proc.add(
        'multiplier', float,
        'The multiplier used to balance the main and random parts of the energy function'
    )
    def params_proc(params):
        pass

    def params_test(params):
        assert params['locations'].shape[1] == center.size
        assert params['standard_deviations'].shape == (params['locations'].shape[0],)

    params = proc.process_params(energy_params, params_proc=params_proc, params_test=params_test)
    n_dim = center.size
    n_bumps = params['locations'].shape[0]
    location = sympy.Array(sympy.symbols('x:{}'.format(n_dim)), (n_dim,))
    center = sympy.Array(center, center.shape)
    r_squared = sympy_array_squared_norm(location - center)
    well_expr = -(energy_params['depth'] / radiuses[1]**4) * (
        r_squared**2 - 2 * radiuses[1]**2 * r_squared
    ) - energy_params['depth']
    mollifier_expr = sympy.functions.exp(
        -radiuses[1] / (radiuses[1] - sympy_array_squared_norm(location - center)**10)
    ) / np.exp(-1)
    random_components = [
        sympy.functions.exp(
            -sympy_array_squared_norm(location - sympy.Array(params['locations'][ii], (n_dim,)))
            / (
                2 * params['standard_deviations'][ii]**2
            )
        ) for ii in range(n_bumps)
    ]
    random_expr = 0
    for ii in range(n_bumps):
        random_expr += random_components[ii]

    sympy_expr = well_expr + params['multiplier'] * mollifier_expr * random_expr
    gradient_expr = sympy.derive_by_array(sympy_expr, location)
    return location, gradient_expr


def generate_random_crater_sympy_expr(center, radiuses, energy_params):
    proc = ParamsProc()
    proc.add(
        'depth', float,
        'The depth of the crater'
    )
    proc.add(
        'height', float,
        'The height of the crater'
    )
    proc.add(
        'locations', np.ndarray,
        'The locations of all the Gaussian random bumps'
    )
    proc.add(
        'standard_deviations', np.ndarray,
        'The standard deviations for the different Gaussian random bumps'
    )
    proc.add(
        'multiplier', float,
        'The multiplier used to balance the main and random parts of the energy function'
    )
    def params_proc(params):
        pass

    def params_test(params):
        assert params['locations'].shape[1] == center.size
        assert params['standard_deviations'].shape == (params['locations'].shape[0],)

    params = proc.process_params(energy_params, params_proc=params_proc, params_test=params_test)
    n_dim = center.size
    n_bumps = params['locations'].shape[0]
    location = sympy.Array(sympy.symbols('x:{}'.format(n_dim)), (n_dim,))
    center = sympy.Array(center, center.shape)
    r_squared = sympy_array_squared_norm(location - center)
    C = 3 * radiuses[1]**2 * sympy.cbrt(
        params['depth'] * params['height'] * (
            params['depth'] + sympy.sqrt(params['depth'] * (params['depth'] + params['height']))
        )
    )
    Delta0 = -9 * params['depth'] * params['height'] * radiuses[1]**4
    b_squared = -(1 / (3 * params['depth'])) * (
        -3 * params['depth'] * radiuses[1]**2 + C + Delta0 / C
    )
    a = params['depth'] / (3 * b_squared * radiuses[1]**4 - radiuses[1]**6)
    crater_expr = a * (
        2 * r_squared**3 - 3 * (b_squared + radiuses[1]**2) * r_squared**2 +
        6 * b_squared * radiuses[1]**2 * r_squared
    ) - energy_params['depth']
    mollifier_expr = sympy.functions.exp(
        -radiuses[1] / (radiuses[1] - sympy_array_squared_norm(location - center)**10)
    ) / np.exp(-1)
    random_components = [
        sympy.functions.exp(
            -sympy_array_squared_norm(location - sympy.Array(params['locations'][ii], (n_dim,)))
            / (
                2 * params['standard_deviations'][ii]**2
            )
        ) for ii in range(n_bumps)
    ]
    random_expr = 0
    for ii in range(n_bumps):
        random_expr += random_components[ii]

    sympy_expr = crater_expr + params['multiplier'] * mollifier_expr * random_expr
    gradient_expr = sympy.derive_by_array(sympy_expr, location)
    return location, gradient_expr


def sympy_array_squared_norm(sympy_array):
    return sympy.tensor.array.tensorcontraction(sympy_array.applyfunc(lambda x: x**2), (0,))


def kfold_split(n_locations, n_fold):
    fold_length = int(np.floor(n_locations / n_fold))
    folds = [range(ii * fold_length, (ii + 1) * fold_length) for ii in range(n_fold - 1)]
    folds.append(range((n_fold - 1) * fold_length, n_locations))
    return folds
