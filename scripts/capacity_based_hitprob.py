import os
import pickle
import tempfile
import timeit

import sacred
from golf_course.core.model import ToyModel
from golf_course.utils import load_model_params
from sacred.observers import FileStorageObserver

log_folder = os.path.expanduser('~/logs/diffusion/capacity_based_hitprob')
ex = sacred.Experiment('capacity_based_hitprob')
ex.observers.append(FileStorageObserver.create(log_folder))


@ex.config
def config():
    case = 'med'
    time_step = 1e-6
    inner_list = [1, 1]
    outer_list = [1, 1]
    num_points_list = [5000, 5000]
    num_clusters_list = [10, 10]
    num_trials_list = [5000, 5000]
    simulation_log_folder = os.path.expanduser(
        '~/entropic_barrier/new_results/simulation_based_hitprob/{}_target'.format(case)
    )
    capacity_estimation_param_list = [
        {
            "inner": inner_list[0],
            "outer": outer_list[0],
            "num_points": num_points_list[0],
            "num_clusters": num_clusters_list[0],
            "num_trials": num_trials_list[0],
            "use_parallel": False,
            "n_split": 1,
            "use_analytical_gradients": True,
            "estimate_gradients": False,
            "n_surfaces_gradients_estimation": 10,
            "time_step": time_step,
        },
        {
            "inner": inner_list[1],
            "outer": outer_list[1],
            "num_points": num_points_list[1],
            "num_clusters": num_clusters_list[1],
            "num_trials": num_trials_list[1],
            "use_parallel": False,
            "n_split": 1,
            "use_analytical_gradients": True,
            "estimate_gradients": False,
            "n_surfaces_gradients_estimation": 10,
            "time_step": time_step,
        },
    ]


@ex.main
def run(simulation_log_folder, time_step, capacity_estimation_param_list):
    temp_folder = tempfile.TemporaryDirectory()
    folder_name = temp_folder.name
    # Construct model
    model_params_fname = os.path.join(simulation_log_folder, 'model_params.pkl')
    ex.add_artifact(model_params_fname)
    _, target_param_list = load_model_params(model_params_fname)
    assert len(capacity_estimation_param_list) == len(target_param_list)
    # Make compatible with existing results
    for ii in range(len(target_param_list)):
        target_param_list[ii] = {
            key: target_param_list[ii][key]
            for key in target_param_list[ii]
            if key not in capacity_estimation_param_list[ii]
        }

    model = ToyModel(time_step, target_param_list)
    # Save simulation results
    results_fname = os.path.join(simulation_log_folder, 'results.pkl')
    ex.add_artifact(results_fname, 'simulation_results.pkl')
    # Estimate capacity-based hitting probability
    start = timeit.default_timer()
    hitting_prob = model.estimate_hitting_prob(capacity_estimation_param_list)
    end = timeit.default_timer()
    time_taken = end - start
    results = {'hitting_prob': hitting_prob, 'time_taken': time_taken}
    print(results)
    results_fname = os.path.join(folder_name, 'capacity_results.pkl')
    with open(results_fname, 'wb') as f:
        pickle.dump(results, f)

    ex.add_artifact(results_fname)
    temp_folder.cleanup()


ex.run(
    config_updates={
        'case': 'large',
        'time_step': 1e-7,
        'outer_list': [5, 5],
        'num_points_list': [1000, 1000],
        'num_clusters_list': [3, 3],
        'num_trials_list': [1000, 1000],
    }
)
ex.run(
    config_updates={
        'case': 'med',
        'time_step': 1e-5,
        'outer_list': [7, 7],
        'num_points_list': [1000, 1000],
        'num_clusters_list': [3, 3],
        'num_trials_list': [1000, 1000],
    }
)
ex.run(
    config_updates={
        'case': 'small',
        'time_step': 1e-6,
        'outer_list': [7, 7],
        'num_points_list': [1000, 1000],
        'num_clusters_list': [3, 3],
        'num_trials_list': [1000, 1000],
    }
)
