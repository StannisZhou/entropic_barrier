import os
import pickle
import tempfile
from pprint import pprint

import numpy as np

import sacred
from golf_course.core.target import Target
from golf_course.estimate.capacity import estimate_capacity
from golf_course.estimate.hitprob import get_simple_hitprob
from sacred.observers import FileStorageObserver

log_folder = os.path.expanduser('~/logs/diffusion/sanity_checks')
ex = sacred.Experiment('sanity_checks')
ex.observers.append(FileStorageObserver.create(log_folder))


@ex.config
def config():
    skip_direction_simulations = False
    n_initial_locations = 100
    n_simulations = 2000
    time_step = 1e-5
    centers = np.array([[0.5, 0.6, 0, 0, 0], [-0.7, 0, 0, 0, 0]])
    radiuses = np.array([[0.02, 0.05, 0.1], [0.04, 0.075, 0.15]])
    capacity_estimation_param = {
        'num_points': int(5e2),
        'time_step': 1e-06,
        'inner': 1,
        'outer': 1,
        'num_clusters': 5,
        'num_trials': int(5e3),
        'use_parallel': False,
        'n_split': 1,
        'use_analytical_gradients': True,
        'estimate_gradients': False,
        "n_surfaces_gradients_estimation": None,
    }


@ex.main
def run(
    skip_direction_simulations,
    n_initial_locations,
    n_simulations,
    time_step,
    centers,
    radiuses,
    capacity_estimation_param,
):
    temp_folder = tempfile.TemporaryDirectory()
    folder_name = temp_folder.name
    results = {}
    if not skip_direction_simulations:
        target_radiuses = np.array([radius[0] for radius in radiuses])
        outer_radiuses = np.array([radius[2] for radius in radiuses])
        initial_location_list, hitting_prob_list, time_taken, expected_hitting_prob = get_simple_hitprob(
            centers,
            target_radiuses,
            outer_radiuses,
            time_step,
            n_initial_locations,
            n_simulations,
        )
        results.update(
            {
                'initial_location_list': initial_location_list,
                'hitting_prob_list': hitting_prob_list,
                'time_taken': time_taken,
                'expected_hitting_prob': expected_hitting_prob,
            }
        )

    target_list = [
        Target(center, radius, 'flat', {}) for center, radius in zip(centers, radiuses)
    ]
    expected_capacity = np.zeros(len(target_list))
    estimated_capacity = np.zeros(len(target_list))
    expected_gradients = np.zeros(len(target_list))
    estimated_gradients = []
    for tt, target in enumerate(target_list):
        n_dim = target.center.size
        estimated_capacity[tt], gradients = estimate_capacity(
            target, **capacity_estimation_param
        )
        estimated_gradients.append(gradients)
        expected_capacity[tt] = (
            target.get_constant()
            * (n_dim - 2)
            / (target.radiuses[0] ** (2 - n_dim) - target.radiuses[2] ** (2 - n_dim))
        )
        expected_gradients[tt] = (n_dim - 2) / (
            target.radiuses[1] ** (n_dim - 1)
            * (target.radiuses[1] ** (2 - n_dim) - target.radiuses[2] ** (2 - n_dim))
        )

    results.update(
        {
            'expected_capacity': expected_capacity,
            'expected_gradients': expected_gradients,
            'estimate_capacity': estimated_capacity,
            'estimated_gradients': estimated_gradients,
        }
    )
    pprint(results)
    results_fname = os.path.join(folder_name, 'results.pkl')
    with open(results_fname, 'wb') as f:
        pickle.dump(results, f)

    ex.add_artifact(results_fname)
    temp_folder.cleanup()


# Large target
radiuses = [[0.15, 0.175, 0.2], [0.15, 0.175, 0.2]]
ex.run(config_updates={'radiuses': radiuses})
# Med target
radiuses = [[0.05, 0.1, 0.2], [0.075, 0.125, 0.2]]
ex.run(config_updates={'radiuses': radiuses})
# Small target
radiuses = [[0.02, 0.05, 0.2], [0.04, 0.075, 0.2]]
ex.run(config_updates={'radiuses': radiuses})
