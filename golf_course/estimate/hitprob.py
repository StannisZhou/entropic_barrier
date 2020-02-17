import functools
import multiprocessing as mp
import timeit

import numpy as np
from tqdm import tqdm

import golf_course.estimate.numba as nestimate
import joblib
from golf_course.utils import sample_uniform_initial_location


def get_simple_hitprob_parallelize(
    centers, target_radiuses, time_step, initial_location_list, n_simulations
):
    n_initial_locations = len(initial_location_list)
    n_targets = centers.shape[0]
    worker = functools.partial(
        nestimate.advance_flat_regions,
        centers=centers,
        radiuses=target_radiuses,
        time_step=time_step,
    )
    hitting_prob_list = np.zeros((n_initial_locations, n_targets))
    if n_simulations == 1:
        # Parallelize over initial locations
        start_time = timeit.default_timer()
        output = joblib.Parallel(n_jobs=joblib.cpu_count())(
            joblib.delayed(worker)(location) for location in tqdm(initial_location_list)
        )
        for ii, (previous_location, current_location, index) in enumerate(output):
            hitting_prob_list[ii][index] = 1

        end_time = timeit.default_timer()
        time_taken = end_time - start_time
    else:
        # Parallelize over simulations
        time_taken = np.zeros(n_initial_locations)
        for ii in range(n_initial_locations):
            initial_location = initial_location_list[ii]
            print('Working on initial location: {}'.format(initial_location))
            start_time = timeit.default_timer()
            indices = []
            with mp.Pool(processes=mp.cpu_count()) as p:
                for previous_location, current_location, index in tqdm(
                    p.imap_unordered(
                        worker, np.tile(initial_location, (n_simulations, 1))
                    )
                ):
                    indices.append(index)

            indices = np.array(indices)
            n_targets = centers.shape[0]
            hitting_prob = np.zeros(n_targets)
            for tt in range(n_targets):
                hitting_prob[tt] = np.sum(indices == tt) / n_simulations

            end_time = timeit.default_timer()
            print(
                'Run {} finished. Hitting probability {}, time taken {}'.format(
                    ii, hitting_prob, end_time - start_time
                )
            )
            hitting_prob_list[ii] = hitting_prob
            time_taken[ii] = end_time - start_time

    return hitting_prob_list, time_taken


def get_simple_hitprob(
    centers,
    target_radiuses,
    outer_radiuses,
    time_step,
    n_initial_locations,
    n_simulations,
):
    # Process parameters
    n_dim = centers.shape[1]
    initial_location_list = np.zeros((n_initial_locations, n_dim))
    for ii in tqdm(range(n_initial_locations)):
        initial_location_list[ii] = sample_uniform_initial_location(
            centers, outer_radiuses, 1.0
        )

    initial_location_list = initial_location_list
    n_targets = centers.shape[0]
    # Ensure parameters are valid
    assert target_radiuses.ndim == 1
    assert target_radiuses.size == n_targets
    assert outer_radiuses.ndim == 1
    assert outer_radiuses.size == n_targets
    assert np.sum(target_radiuses > 0) == n_targets
    for ii in range(n_targets):
        assert np.linalg.norm(centers[ii]) + outer_radiuses[ii] < 1
        assert outer_radiuses[ii] > target_radiuses[ii]

    for ii in range(n_targets - 1):
        for jj in range(ii + 1, n_targets):
            assert (
                np.linalg.norm(centers[ii] - centers[jj])
                > outer_radiuses[ii] + outer_radiuses[jj]
            )
    expected_hitting_prob = (
        1 / (target_radiuses ** (2 - n_dim) - outer_radiuses ** (2 - n_dim))
    ) / np.sum(1 / (target_radiuses ** (2 - n_dim) - outer_radiuses ** (2 - n_dim)))
    hitting_prob_list, time_taken = get_simple_hitprob_parallelize(
        centers, target_radiuses, time_step, initial_location_list, n_simulations
    )
    return initial_location_list, hitting_prob_list, time_taken, expected_hitting_prob


def get_nontrivial_hitprob(toy_model, n_initial_locations, n_simulations):
    n_targets = len(toy_model.target_list)
    n_dim = toy_model.target_list[0].center.size
    initial_location_list = np.zeros((n_initial_locations, n_dim))
    centers = np.zeros((n_targets, n_dim))
    outer_radiuses = np.zeros(n_targets)
    for ii in range(n_targets):
        centers[ii] = toy_model.target_list[ii].center
        outer_radiuses[ii] = toy_model.target_list[ii].radiuses[2]

    for ii in tqdm(range(n_initial_locations)):
        initial_location_list[ii] = sample_uniform_initial_location(
            centers, outer_radiuses, 1.0
        )

    n_targets = len(toy_model.target_list)
    hitting_prob_list = np.zeros((n_initial_locations, n_targets))
    if n_simulations == 1:
        # Parallelize over initial locations
        start_time = timeit.default_timer()
        output = joblib.Parallel(n_jobs=joblib.cpu_count())(
            joblib.delayed(toy_model.do_naive_simulation)(location)
            for location in tqdm(initial_location_list)
        )
        for ii, index in enumerate(output):
            hitting_prob_list[ii][index] = 1

        end_time = timeit.default_timer()
        time_taken = end_time - start_time
    else:
        # Parallelize over simulations
        time_taken = np.zeros(n_initial_locations)
        for run_idx in range(n_initial_locations):
            initial_location = initial_location_list[run_idx]
            print('Working on initial location: {}'.format(initial_location))
            indices = []
            with mp.Pool(processes=mp.cpu_count()) as p:
                for index in tqdm(
                    p.imap_unordered(
                        toy_model.do_naive_simulation,
                        np.tile(initial_location, (n_simulations, 1)),
                    )
                ):
                    indices.append(index)

            indices = np.array(indices)
            hitting_prob = np.zeros(n_targets)
            for target_idx in range(n_targets):
                hitting_prob[target_idx] = np.sum(indices == target_idx) / n_simulations

            end_time = timeit.default_timer()
            print(
                'Run {} finished. Hitting probability {}, time taken {}'.format(
                    run_idx, hitting_prob, end_time - start_time
                )
            )
            hitting_prob_list[run_idx] = hitting_prob
            time_taken[run_idx] = end_time - start_time

    return initial_location_list, hitting_prob_list, time_taken
