import os
import pickle
import tempfile

import numpy as np

import sacred
from golf_course.core.model import ToyModel
from golf_course.estimate.hitprob import get_nontrivial_hitprob
from golf_course.utils import (DEFAULT_RELATIVE_SCALE, load_model_params,
                               sample_random_locations)
from sacred.observers import FileStorageObserver

log_folder = os.path.expanduser('~/logs/diffusion/simulation_based_hitprob')
ex = sacred.Experiment('simulation_based_hitprob')
ex.observers.append(FileStorageObserver.create(log_folder))


def generate_model_params(n_bumps=10, relative_scale=DEFAULT_RELATIVE_SCALE):
    time_step = 1e-05
    centers = np.array([[0.5, 0.6, 0, 0, 0], [-0.7, 0, 0, 0, 0]])
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    target_param_list = [
        {
            "center": centers[0],
            "radiuses": np.array([0.02, 0.05, 0.1]),
            "energy_type": "random_well",
            "energy_params": {
                "depth": 10.0,
                "locations": sample_random_locations(
                    np.array([0.5, 0.6, 0, 0, 0]), np.array([0.02, 0.05]), n_bumps
                ),
                "standard_deviations": 0.01 * np.ones(n_bumps),
            },
        },
        {
            "center": centers[1],
            "radiuses": np.array([0.04, 0.075, 0.15]),
            "energy_type": "random_crater",
            "energy_params": {
                "depth": 6.0,
                "height": 1.0,
                "locations": sample_random_locations(
                    np.array([-0.7, 0, 0, 0, 0]), np.array([0.04, 0.075]), n_bumps
                ),
                "standard_deviations": 0.01 * np.ones(n_bumps),
            },
        },
    ]
    for target_param in target_param_list:
        if target_param['energy_type'] == 'random_well':
            energy_range = target_param['energy_params']['depth']
        elif target_param['energy_type'] == 'random_crater':
            energy_range = (
                target_param['energy_params']['depth']
                + target_param['energy_params']['height']
            )
        else:
            assert False, 'Wrong energy type.'

        target_param['energy_params']['multiplier'] = float(
            (relative_scale * energy_range)
        )

    return time_step, target_param_list


@ex.config
def config():
    n_initial_locations = 2
    n_simulations = 2000
    model_params_fname = None
    if model_params_fname is None:
        time_step, target_param_list = generate_model_params()
    else:
        time_step, target_param_list = load_model_params(model_params_fname)


@ex.main
def run(n_initial_locations, n_simulations, time_step, target_param_list):
    temp_folder = tempfile.TemporaryDirectory()
    folder_name = temp_folder.name
    model_params = {'time_step': time_step, 'target_param_list': target_param_list}
    model_params_fname = os.path.join(folder_name, 'model_params.pkl')
    with open(model_params_fname, 'wb') as f:
        pickle.dump(model_params, f)

    ex.add_artifact(model_params_fname)
    model = ToyModel(time_step, target_param_list)
    initial_location_list, hitting_prob_list, time_taken = get_nontrivial_hitprob(
        model, n_initial_locations, n_simulations
    )
    results = {
        'n_initial_locations': n_initial_locations,
        'n_simulations': n_simulations,
        'initial_location_list': initial_location_list,
        'hitting_prob_list': hitting_prob_list,
        'time_taken': time_taken,
    }
    results_fname = os.path.join(folder_name, 'results.pkl')
    with open(results_fname, 'wb') as f:
        pickle.dump(results, f)

    ex.add_artifact(results_fname)
    temp_folder.cleanup()


ex.run()
