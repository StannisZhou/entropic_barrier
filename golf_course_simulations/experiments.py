import numpy as np
import h5py
import logging
from golf_course_simulations.basic_components import ThreeSphere, ToyModel
from golf_course_simulations.utils import ParamsProc, _sample_uniform_initial_location, uniform_on_sphere
import golf_course_simulations.numba_support as ns
from tqdm import tqdm
import functools
import multiprocessing as mp
import timeit
import pickle
import shutil
import os


class SimpleHittingProbTest(object):
    @staticmethod
    def get_proc():
        proc = ParamsProc()
        proc.add(
            'root_folder', str,
            'The path to the root folder in which we are running the experiments'
        )
        proc.add(
            'output_identifier', str,
            'The output identifier, which we are going to use as the folder name for storing all the outputs'
        )
        proc.add(
            'centers', np.ndarray,
            'The centers of the three spheres. Each row corresponds to a center'
        )
        proc.add(
            'target_radiuses', np.ndarray,
            'The radiuses of each one of the targets'
        )
        proc.add(
            'outer_radiuses', np.ndarray,
            'The radiuses outside which we are going to sample the initial_locations'
        )
        proc.add(
            'time_step', float,
            'The time step we are going to use for the simulations'
        )
        proc.add(
            'n_initial_locations', int,
            'The number of initial locations we are going to test'
        )
        proc.add(
            'n_simulations', int,
            'The number of simulations we are going to run to estimate the hitting probabilities'
        )
        return proc

    def __init__(self, params):
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(name)s %(levelname)s: %(message)s')
        console.setFormatter(formatter)
        logger = logging.getLogger(type(self).__name__)
        logger.addHandler(console)
        self.logger = logger
        proc = self.get_proc()
        def params_proc(params):
            params['realized_params_fname'] = '{}/output/{}/realized_params.pkl'.format(
                params['root_folder'], params['output_identifier']
            )
            params['hitting_prob_fname'] = '{}/output/{}/hitting_prob.hdf5'.format(
                params['root_folder'], params['output_identifier']
            )
            self.logger.info('Sampling initial locations')
            n_dim = params['centers'].shape[1]
            initial_location_list = np.zeros((params['n_initial_locations'], n_dim))
            for ii in tqdm(range(params['n_initial_locations'])):
                initial_location_list[ii] = _sample_uniform_initial_location(
                    params['centers'], params['outer_radiuses'], 1.0
                )

            params['initial_location_list'] = initial_location_list

        def params_test(params):
            n_targets = params['centers'].shape[0]
            n_dim = params['centers'].shape[1]
            assert params['target_radiuses'].ndim == 1
            assert params['target_radiuses'].size == n_targets
            assert params['outer_radiuses'].ndim == 1
            assert params['outer_radiuses'].size == n_targets
            assert np.sum(params['target_radiuses'] > 0) == n_targets
            for ii in range(n_targets):
                assert np.linalg.norm(params['centers'][ii]) + params['outer_radiuses'][ii] < 1
                assert params['outer_radiuses'][ii] > params['target_radiuses'][ii]

            for ii in range(n_targets - 1):
                for jj in range(ii + 1, n_targets):
                    assert np.linalg.norm(
                        params['centers'][ii] - params['centers'][jj]
                    ) > params['outer_radiuses'][ii] + params['outer_radiuses'][jj]

        self.logger.info('Initializing class')
        self.params = proc.process_params(params, params_proc, params_test)
        with open(self.params['realized_params_fname'], 'wb') as f:
            pickle.dump(self.params, f)

    def run_experiments(self):
        n_targets = self.params['centers'].shape[0]
        n_dim = self.params['centers'].shape[1]
        target_radiuses = self.params['target_radiuses']
        outer_radiuses = self.params['outer_radiuses']
        expected_harmonic_hitting_prob = target_radiuses**(n_dim - 2) / np.sum(target_radiuses**(n_dim - 2))
        expected_condenser_hitting_prob = (
            1 / (target_radiuses**(2 - n_dim) - outer_radiuses**(2 - n_dim))
        ) / np.sum(
            1 / (target_radiuses**(2 - n_dim) - outer_radiuses**(2 - n_dim))
        )
        self.logger.info(
            'Expected hitting prob based on harmonic capacity: {}'.format(
                expected_harmonic_hitting_prob
            )
        )
        self.logger.info(
            'Expected hitting prob based on condenser capacity: {}'.format(
                expected_condenser_hitting_prob
            )
        )
        with h5py.File(self.params['hitting_prob_fname'], 'w') as f:
            f['expected_harmonic_hitting_prob'] = expected_harmonic_hitting_prob
            f['expected_condenser_hitting_prob'] = expected_condenser_hitting_prob

        hitting_prob_list = np.zeros((self.params['n_initial_locations'], n_targets))
        time_taken = np.zeros(self.params['n_initial_locations'])
        for ii in range(self.params['n_initial_locations']):
            self.logger.info(
                'Working on initial location: {}'.format(
                    self.params['initial_location_list'][ii]
                )
            )
            start_time = timeit.default_timer()
            hitting_prob = self.run_one_test(self.params['initial_location_list'][ii], n_targets)
            end_time = timeit.default_timer()
            self.logger.info(
                'Run {} finished. Hitting probability {}, time taken {}'.format(
                    ii, hitting_prob, end_time - start_time
                )
            )
            hitting_prob_list[ii] = hitting_prob
            time_taken[ii] = end_time - start_time
            with h5py.File(self.params['hitting_prob_fname'], 'a') as f:
                if 'hitting_prob_list' in f:
                    del f['hitting_prob_list']

                if 'time_taken' in f:
                    del f['time_taken']

                f['hitting_prob_list'] = hitting_prob_list[:ii + 1]
                f['time_taken'] = time_taken[:ii + 1]

    def run_one_test(self, initial_location, n_targets):
        indices = []
        worker = functools.partial(
            ns.advance_flat_regions, centers=self.params['centers'],
            radiuses=self.params['target_radiuses'], time_step=self.params['time_step']
        )
        with mp.Pool(processes=mp.cpu_count()) as p:
            for previous_location, current_location, index in tqdm(
                p.imap_unordered(
                    worker,
                    np.tile(initial_location, (self.params['n_simulations'], 1)))
            ):
                indices.append(index)

        indices = np.array(indices)
        hitting_prob = np.zeros(n_targets)
        for ii in range(n_targets):
            hitting_prob[ii] = np.sum(indices == ii) / self.params['n_simulations']

        return hitting_prob


class CapacityEstimationTest(object):
    @staticmethod
    def get_proc():
        proc = ParamsProc()
        proc.update(ThreeSphere.get_proc())
        proc.add(
            'root_folder', str,
            'The path to the root folder in which we are running the experiments'
        )
        proc.add(
            'output_identifier', str,
            'The output identifier, which we are going to use as the folder name for storing all the outputs'
        )
        return proc

    def __init__(self, params):
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(name)s %(levelname)s: %(message)s')
        console.setFormatter(formatter)
        logger = logging.getLogger(type(self).__name__)
        logger.addHandler(console)
        self.logger = logger
        proc = self.get_proc()
        def params_proc(params):
            params['realized_params_fname'] = '{}/output/{}/realized_params.pkl'.format(
                params['root_folder'], params['output_identifier']
            )
            params['results_fname'] = '{}/output/{}/results.hdf5'.format(
                params['root_folder'], params['output_identifier']
            )

        def params_test(params):
            assert params['energy_type'] == 'flat'

        self.logger.info('Initializing class')
        self.params = proc.process_params(params, params_proc, params_test)
        with open(self.params['realized_params_fname'], 'wb') as f:
            pickle.dump(self.params, f)

        three_sphere_param  = {
            key: self.params[key] for key in set(self.params.keys()).intersection(
                ThreeSphere.get_proc().keys
            )
        }
        self.three_sphere = ThreeSphere(three_sphere_param)
        self.three_sphere.generate_force_field_function()

    def run_experiments(self):
        n_dim = self.params['center'].size
        radiuses = self.params['radiuses']
        expected_capacity = (n_dim - 2) / (radiuses[0]**(2 - n_dim) - radiuses[2]**(2 - n_dim))
        self.logger.info('Expected capacity {}'.format(expected_capacity))
        capacity = self.three_sphere.estimate_capacity()
        self.logger.info('Estimated capacity {}'.format(capacity))
        with h5py.File(self.params['results_fname'], 'w') as f:
            f['expected_capacity'] = expected_capacity
            f['estimated_capacity'] = capacity


class NontrivialHittingProbTest(object):
    @staticmethod
    def get_proc():
        proc = ParamsProc()
        proc.add(
            'root_folder', str,
            'The path to the root folder in which we are running the experiments'
        )
        proc.add(
            'output_identifier', str,
            'The output identifier, which we are going to use as the folder name for storing all the outputs'
        )
        proc.add(
            'n_initial_locations', int,
            'The number of initial locations we are going to test'
        )
        proc.add(
            'n_simulations', int,
            'The number of simulations we are going to run to estimate the hitting probabilities'
        )
        proc.update(ToyModel.get_proc())
        return proc

    def __init__(self, params):
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(name)s %(levelname)s: %(message)s')
        console.setFormatter(formatter)
        logger = logging.getLogger(type(self).__name__)
        logger.addHandler(console)
        self.logger = logger
        proc = self.get_proc()
        def params_proc(params):
            params['realized_params_fname'] = '{}/output/{}/realized_params.pkl'.format(
                params['root_folder'], params['output_identifier']
            )
            params['hitting_prob_fname'] = '{}/output/{}/hitting_prob.hdf5'.format(
                params['root_folder'], params['output_identifier']
            )
            self.logger.info('Sampling initial locations')
            n_three_spheres = len(params['three_sphere_param_list'])
            n_dim = params['three_sphere_param_list'][0]['center'].size
            initial_location_list = np.zeros((params['n_initial_locations'], n_dim))
            centers = np.zeros((n_three_spheres, n_dim))
            outer_radiuses = np.zeros(n_three_spheres)
            for ii in range(n_three_spheres):
                centers[ii] = params['three_sphere_param_list'][ii]['center']
                outer_radiuses[ii] = params['three_sphere_param_list'][ii]['radiuses'][2]

            for ii in tqdm(range(params['n_initial_locations'])):
                initial_location_list[ii] = _sample_uniform_initial_location(
                    centers, outer_radiuses, 1.0
                )

            params['initial_location_list'] = initial_location_list

        def params_test(params):
            pass

        self.logger.info('Initializing class')
        self.params = proc.process_params(params, params_proc, params_test)
        with open(self.params['realized_params_fname'], 'wb') as f:
            pickle.dump(self.params, f)

        self.logger.info('Initializing toy model')
        params = {
            key: self.params[key] for key in set(self.params.keys()).intersection(
                ToyModel.get_proc().keys
            )
        }
        self.toy_model = ToyModel(params)

    def run_experiments(self):
        n_targets = len(self.params['three_sphere_param_list'])
        hitting_prob_list = np.zeros((self.params['n_initial_locations'], n_targets))
        time_taken = np.zeros(self.params['n_initial_locations'])
        for ii in range(self.params['n_initial_locations']):
            self.logger.info(
                'Working on initial location: {}'.format(
                    self.params['initial_location_list'][ii]
                )
            )
            start_time = timeit.default_timer()
            hitting_prob = self.run_one_test(self.params['initial_location_list'][ii], n_targets)
            end_time = timeit.default_timer()
            self.logger.info(
                'Run {} finished. Hitting probability {}, time taken {}'.format(
                    ii, hitting_prob, end_time - start_time
                )
            )
            hitting_prob_list[ii] = hitting_prob
            time_taken[ii] = end_time - start_time
            with h5py.File(self.params['hitting_prob_fname'], 'a') as f:
                if 'hitting_prob_list' in f:
                    del f['hitting_prob_list']

                if 'time_taken' in f:
                    del f['time_taken']

                if 'n_cpus' in f:
                    del f['n_cpus']

                f['hitting_prob_list'] = hitting_prob_list[:ii + 1]
                f['time_taken'] = time_taken[:ii + 1]
                f['n_cpus'] = mp.cpu_count()


    def run_one_test(self, initial_location, n_targets):
        if self.params['n_simulations'] == 1:
            index = self.toy_model.do_naive_simulation(initial_location)
            indices = np.array([index])
        else:
            indices = []
            with mp.Pool(processes=mp.cpu_count()) as p:
                for index in tqdm(p.imap_unordered(
                    self.toy_model.do_naive_simulation, np.tile(initial_location, (self.params['n_simulations'], 1)))
                ):
                    indices.append(index)

            indices = np.array(indices)

        hitting_prob = np.zeros(n_targets)
        for ii in range(n_targets):
            hitting_prob[ii] = np.sum(indices == ii) / self.params['n_simulations']

        return hitting_prob


class HittingProbEstimationCapacity(object):
    @staticmethod
    def get_proc():
        proc = ParamsProc()
        proc.add(
            'root_folder', str,
            'The path to the root folder in which we are running the experiments'
        )
        proc.add(
            'output_identifier', str,
            'The output identifier, which we are going to use as the folder name for storing all the outputs'
        )
        proc.add(
            'time_step', float,
            'The time step we are going to use for the simulation',
            1e-5
        )
        proc.add(
            'naive_simulation_folder', str,
            'The folder containing naive simulation results'
        )
        proc.add(
            'capacity_estimator_params', list,
            'A list containing relevant parameters for the capacity estimator for each three sphere'
        )
        return proc

    def __init__(self, params):
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(name)s %(levelname)s: %(message)s')
        console.setFormatter(formatter)
        logger = logging.getLogger(type(self).__name__)
        logger.addHandler(console)
        self.logger = logger
        proc = self.get_proc()
        def params_proc(params):
            params['realized_params_fname'] = '{}/output/{}/realized_params.pkl'.format(
                params['root_folder'], params['output_identifier']
            )
            params['results_fname'] = '{}/output/{}/results.hdf5'.format(
                params['root_folder'], params['output_identifier']
            )
            params['naive_simulation_params_fname'] = '{}/realized_params.pkl'.format(
                params['naive_simulation_folder']
            )
            with open(params['naive_simulation_params_fname'], 'rb') as f:
                naive_simulation_params = pickle.load(f)

            self.logger.info('Moving the naive simulation results')
            shutil.copytree(
                params['naive_simulation_folder'], '{}/output/{}/{}'.format(
                    params['root_folder'], params['output_identifier'],
                    os.path.basename(params['naive_simulation_folder'])
                )
            )
            params['three_sphere_param_list'] = naive_simulation_params['three_sphere_param_list']
            n_three_spheres = len(params['three_sphere_param_list'])
            for ii in range(n_three_spheres):
                params['three_sphere_param_list'][ii].update(params['capacity_estimator_params'][ii])

        def params_test(params):
            assert len(params['three_sphere_param_list']) == len(params['capacity_estimator_params'])

        self.logger.info('Initializing class')
        self.params = proc.process_params(params, params_proc, params_test)
        with open(self.params['realized_params_fname'], 'wb') as f:
            pickle.dump(self.params, f)

        self.logger.info('Initializing toy model')
        params = {
            key: self.params[key] for key in set(self.params.keys()).intersection(
                ToyModel.get_proc().keys
            )
        }
        self.toy_model = ToyModel(params)

    def run_experiments(self):
        start = timeit.default_timer()
        hitting_prob = self.toy_model.estimate_hitting_prob()
        end = timeit.default_timer()
        time_taken = end - start
        self.logger.info('Hitting prob estimation finished in {}s. Hitting prob: {}'.format(
            time_taken, hitting_prob
        ))
        with h5py.File(self.params['results_fname'], 'w') as f:
            f['hitting_prob'] = hitting_prob
            f['time_taken'] = time_taken
