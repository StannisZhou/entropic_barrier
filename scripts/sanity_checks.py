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
    do_direct_simulations = True
    do_capacity_estimation = True
    do_gradients_estimation = False
    n_initial_locations = 100
    n_simulations = 2000
    time_step = 1e-5
    centers = [[0.5, 0.6, 0, 0, 0], [-0.7, 0, 0, 0, 0]]
    radiuses = [[0.02, 0.05, 0.1], [0.04, 0.075, 0.15]]
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
        'estimate_gradients': do_gradients_estimation,
        "n_surfaces_gradients_estimation": None,
    }


@ex.main
def run(
    do_direct_simulations,
    do_capacity_estimation,
    do_gradients_estimation,
    n_initial_locations,
    n_simulations,
    time_step,
    centers,
    radiuses,
    capacity_estimation_param,
):
    centers = np.array(centers)
    radiuses = np.array(radiuses)
    if do_gradients_estimation:
        assert do_capacity_estimation

    temp_folder = tempfile.TemporaryDirectory()
    folder_name = temp_folder.name
    results = {}
    if do_direct_simulations:
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

    if do_capacity_estimation:
        target_list = [
            Target(center, radius, 'flat', {})
            for center, radius in zip(centers, radiuses)
        ]
        expected_capacity = np.zeros(len(target_list))
        estimated_capacity = np.zeros(len(target_list))
        if do_gradients_estimation:
            expected_gradients = np.zeros(len(target_list))
            estimated_gradients = np.zeros(
                (len(target_list), capacity_estimation_param['num_clusters'])
            )

        for tt, target in enumerate(target_list):
            n_dim = target.center.size
            estimated_capacity[tt], gradients = estimate_capacity(
                target, **capacity_estimation_param
            )
            expected_capacity[tt] = (
                target.get_constant()
                * (n_dim - 2)
                / (
                    target.radiuses[0] ** (2 - n_dim)
                    - target.radiuses[2] ** (2 - n_dim)
                )
            )
            if do_gradients_estimation:
                estimated_gradients[tt] = gradients
                expected_gradients[tt] = (n_dim - 2) / (
                    target.radiuses[1] ** (n_dim - 1)
                    * (
                        target.radiuses[1] ** (2 - n_dim)
                        - target.radiuses[2] ** (2 - n_dim)
                    )
                )

        results.update(
            {
                'expected_capacity': expected_capacity,
                'estimate_capacity': estimated_capacity,
            }
        )
        if do_gradients_estimation:
            results.update(
                {
                    'expected_gradients': expected_gradients,
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
# Accurate estimate of mean hitting probabilities
n_initial_locations = 50000
n_simulations = 1
# Large target
radiuses = [[0.15, 0.175, 0.2], [0.15, 0.175, 0.2]]
ex.run(
    config_updates={
        'n_initial_locations': n_initial_locations,
        'n_simulations': n_simulations,
        'radiuses': radiuses,
        'do_capacity_estimation': False,
    }
)
# Med target
radiuses = [[0.05, 0.1, 0.2], [0.075, 0.125, 0.2]]
ex.run(
    config_updates={
        'n_initial_locations': n_initial_locations,
        'n_simulations': n_simulations,
        'radiuses': radiuses,
        'do_capacity_estimation': False,
    }
)
# Small target
radiuses = [[0.02, 0.05, 0.2], [0.04, 0.075, 0.2]]
ex.run(
    config_updates={
        'n_initial_locations': n_initial_locations,
        'n_simulations': n_simulations,
        'radiuses': radiuses,
        'do_capacity_estimation': False,
    }
)
