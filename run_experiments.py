import numpy as np
from golf_course_simulations.experiments import SimpleHittingProbTest, CapacityEstimationTest
from golf_course_simulations.experiments import NontrivialHittingProbTest, HittingProbEstimationCapacity
from golf_course_simulations.utils import sample_random_locations
import os
import shutil
import logging


def reset_logging(output_identifier):
    logs_fname = '{}/{}/logs.txt'.format(output_folder, output_identifier)
    logs_level = logging.INFO
    if not os.path.exists(os.path.dirname(logs_fname)):
        os.makedirs(os.path.dirname(logs_fname))

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logs_level,
        format='%(asctime)s %(name)s %(levelname)s: %(message)s',
        datefmt='%d%m%y-%H%M%S',
        filename=logs_fname,
        filemode='w'
    )


ROOT_FOLDER = os.path.dirname(os.path.abspath(__file__))
output_folder = '{}/output'.format(ROOT_FOLDER)
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

#  Hitting probabilities for flat energy landscape
output_identifier = 'simple_hitting_prob_test'
reset_logging(output_identifier)
params = {
    'target_radiuses': np.array([0.02, 0.04]),
    'n_initial_locations': 100,
    'n_simulations': 2000,
    'centers': np.array([
        [0.5, 0.6, 0, 0, 0],
        [-0.7, 0, 0, 0, 0]
    ]),
    'time_step': 1e-5,
    'outer_radiuses': np.array([0.1, 0.15]),
    'output_identifier': output_identifier,
    'root_folder': ROOT_FOLDER
}
obj = SimpleHittingProbTest(params)
obj.run_experiments()

# Capacity estimation algorithm for flat energy landscape
output_identifier = 'capacity_estimation_test'
reset_logging(output_identifier)
params = {
    'energy_type': 'flat',
    'radiuses': np.array([0.02, 0.05, 0.1]),
    'num_points': int(1e2),
    'time_step': 1e-06,
    'inner': 1,
    'num_clusters': 3,
    'relative_scale': 0.1,
    'center': np.zeros(5),
    'outer': 1,
    'num_trials': int(1e3),
    'energy_params': {},
    'use_parallel': False,
    'n_split': 1,
    'output_identifier': output_identifier,
    'root_folder': ROOT_FOLDER
}
obj = CapacityEstimationTest(params)
obj.run_experiments()

# Hitting probabilities for nontrivial energy landscape
output_identifier = 'nontrivial_hitting_prob_test'
reset_logging(output_identifier)
n_bumps = 10
params = {
    'n_initial_locations': 100,
    'n_simulations': 2000,
    'time_step': 1e-05,
    'three_sphere_param_list': [
        {
            'center': np.array([0.5, 0.6, 0, 0, 0]),
            'radiuses': np.array([0.02, 0.05, 0.1]),
            'energy_type': 'random_well',
            'energy_params': {
                'depth': 10.,
                "locations": sample_random_locations(np.array([0.5, 0.6, 0, 0, 0]), np.array([0.02, 0.05]), n_bumps),
                "standard_deviations": 0.01 * np.ones(n_bumps)
            },
            'inner': 2,
            'outer': 2,
            'num_points': 10000,
            'num_clusters': 15,
            'num_trials': 5000
        },
        {
            'center': np.array([-0.7, 0, 0, 0, 0]),
            'radiuses': np.array([0.04, 0.075, 0.15]),
            'energy_type': 'random_crater',
            'energy_params': {
                'depth': 6.,
                'height': 1.0,
                "locations": sample_random_locations(np.array([-0.7, 0, 0, 0, 0]), np.array([0.04, 0.075]), n_bumps),
                "standard_deviations": 0.01 * np.ones(n_bumps)
            },
            'inner': 2,
            'outer': 2,
            'num_points': 10000,
            'num_clusters': 15,
            'num_trials': 5000
        }
    ],
    'output_identifier': output_identifier,
    'root_folder': ROOT_FOLDER
}
obj = NontrivialHittingProbTest(params)
obj.run_experiments()

# CHop estimate
output_identifier = 'hitting_prob_estimation_capacity'
reset_logging(output_identifier)
params = {
    'naive_simulation_folder': '{}/nontrivial_hitting_prob_test'.format(output_folder),
    'time_step': 1e-6,
    'capacity_estimator_params': [
        {
            'inner': 1,
            'outer': 1,
            'num_points': 3000,
            'num_clusters': 5,
            'num_trials': 1000,
            'use_parallel': False,
            'n_split': 1
        },
        {
            'inner': 1,
            'outer': 2,
            'num_points': 3000,
            'num_clusters': 5,
            'num_trials': 1000,
            'use_parallel': False,
            'n_split': 1
        }
    ],
    'output_identifier': output_identifier,
    'root_folder': ROOT_FOLDER
}
obj = HittingProbEstimationCapacity(params)
obj.run_experiments()
