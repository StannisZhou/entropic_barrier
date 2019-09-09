import multiprocessing as mp

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

import golf_course.estimate.numba as nestimate
from golf_course.utils import uniform_on_sphere
from tqdm import tqdm


def estimate_capacity(
    target,
    inner,
    outer,
    num_points,
    num_clusters,
    num_trials,
    time_step=1e-5,
    use_parallel=True,
    n_split=4,
    use_analytical_gradients=True,
    estimate_gradients=False,
    n_surfaces_gradients_estimation=15,
):
    """
    Parameters
    ----------
    inner: int
        The number of intermediate layers we are going to use for calculating the inner rate
    outer: int
        The number of intermediate layers we are going to use for calculating the outer rate
    num_points: int
        The number of points we are going to have on each layer.  We are going to form clusters
        based on these points.
    num_clusters: int
        The number of clusters we are going to have on each layer
    num_trials: int
        The number of trials we are going to run for each bin in order to decide the transition probabilities
    time_step : float
        The time step we are going to use for the simulation. Default to 1e-5
    use_parallel : bool
        Whether we are going to make the code parallel or not
    n_split : int
        The number of splits we are going to use for making things parallel. Default to 4
    n_surfaces_gradients_estimation : int
        The number of surfaces we are going to use for numerically estimating the gradients
    analytical_gradients : bool
        Whether we want to use the gradients estimated analytically

    """
    if not use_analytical_gradients:
        assert estimate_gradients

    hitting_prob, cluster_centers, cluster_labels = estimate_hitting_prob(
        target,
        target.radiuses,
        inner,
        outer,
        num_points,
        num_clusters,
        num_trials,
        time_step,
        use_parallel,
        n_split,
    )
    middle_index = outer + 1
    cluster_labels = cluster_labels[middle_index]
    cluster_centers = cluster_centers[middle_index]
    n_points_in_clusters = np.array(
        [np.sum(cluster_labels == ii) for ii in range(num_clusters)]
    )
    n_dim = target.center.size
    dA = target.radiuses[1]
    if estimate_gradients:
        delta = (target.radiuses[2] - target.radiuses[1]) / (
            n_surfaces_gradients_estimation + 2
        )
        radiuses_gradients_estimation = np.array(
            [target.radiuses[1], target.radiuses[1] + delta, target.radiuses[2]]
        )
        hitting_prob_gradients, cluster_centers_gradients, cluster_labels_gradients = estimate_hitting_prob(
            target,
            radiuses_gradients_estimation,
            0,
            n_surfaces_gradients_estimation,
            num_points,
            num_clusters,
            num_trials,
            time_step,
            use_parallel,
            n_split,
        )
        cluster_centers_gradients = cluster_centers_gradients[
            n_surfaces_gradients_estimation + 1
        ]
        _, ind = linear_sum_assignment(
            cdist(cluster_centers, cluster_centers_gradients)
        )
        hitting_prob_gradients = hitting_prob_gradients[ind]
        gradients = np.abs(hitting_prob_gradients - 1) / delta
    else:
        gradients = None

    if use_analytical_gradients:
        rAtilde = target.radiuses[2]
        capacity = (
            (n_dim - 2)
            / (dA ** (2 - n_dim) - rAtilde ** (2 - n_dim))
            * np.sum(n_points_in_clusters * hitting_prob)
            / num_points
        )
    else:
        capacity = (
            dA ** (n_dim - 1)
            * np.sum(n_points_in_clusters * hitting_prob * gradients)
            / num_points
        )

    capacity *= target.get_constant()
    return capacity, gradients


def estimate_hitting_prob(
    target,
    radiuses,
    inner,
    outer,
    num_points,
    num_clusters,
    num_trials,
    time_step,
    use_parallel,
    n_split,
):
    cluster_centers, cluster_labels, propagated_points, statistics_from_propagation = _propagate_and_cluster(
        target, radiuses, inner, outer, num_points, num_clusters, time_step
    )
    forward_probabilities, backward_probabilities, cluster_labels = _get_data_driven_binning_transition_probabilities(
        target,
        radiuses,
        inner,
        outer,
        num_clusters,
        num_trials,
        time_step,
        use_parallel,
        n_split,
        cluster_centers,
        cluster_labels,
        propagated_points,
        statistics_from_propagation,
    )
    print('Transition probabilities calculation done.')
    hitting_prob = _get_data_driven_binning_hitting_probability(
        forward_probabilities, backward_probabilities, inner, outer, num_clusters
    )
    return hitting_prob, cluster_centers, cluster_labels


def _get_data_driven_binning_transition_probabilities(
    target,
    radiuses,
    inner,
    outer,
    num_clusters,
    num_trials,
    time_step,
    use_parallel,
    n_split,
    cluster_centers,
    cluster_labels,
    propagated_points,
    statistics_from_propagation,
):
    forward_probabilities = []
    backward_probabilities = []
    forward_probabilities, backward_probabilities = _additional_simulations_for_transition_probabilities(
        target,
        radiuses,
        cluster_centers,
        cluster_labels,
        propagated_points,
        statistics_from_propagation,
        inner,
        outer,
        num_clusters,
        num_trials,
        time_step,
        use_parallel,
        n_split,
    )
    return forward_probabilities, backward_probabilities, cluster_labels


def _propagate_and_cluster(
    target, radiuses, inner, outer, num_points, num_clusters, time_step
):
    center = target.center
    initial_locations = uniform_on_sphere(
        center, radiuses[1], num_samples=num_points, reflecting_boundary_radius=1
    )
    num_surfaces = inner + outer + 3
    middle_index = outer + 1
    surfaces = _get_surfaces(radiuses, inner, outer)
    assert len(surfaces) == num_surfaces, 'The generated surfaces are not right.'
    # Propagate the points and gather information
    propagated_points = [[] for _ in range(num_surfaces)]
    propagated_points[middle_index] = initial_locations
    propagated_information = []
    extra_information = []
    print('Doing propagation.')
    # Do the initial propagation from the middle sphere
    _propagate_and_get_info(
        target,
        surfaces,
        propagated_points,
        propagated_information,
        extra_information,
        middle_index,
        num_points,
        time_step,
    )
    # Do the forward propagation, from the middle sphere to the inner sphere
    for index in range(middle_index + 1, num_surfaces - 1):
        _propagate_and_get_info(
            target,
            surfaces,
            propagated_points,
            propagated_information,
            extra_information,
            index,
            num_points,
            time_step,
        )
    # Do the backward propagation, from the middle sphere to the outer sphere
    for index in range(middle_index - 1, 0, -1):
        _propagate_and_get_info(
            target,
            surfaces,
            propagated_points,
            propagated_information,
            extra_information,
            index,
            num_points,
            time_step,
        )
    # Do the clustering
    cluster_centers = [[] for _ in range(num_surfaces)]
    cluster_labels = [[] for _ in range(num_surfaces)]
    print('Doing clustering.')
    for ii in tqdm(range(num_surfaces)):
        cluster_centers[ii], cluster_labels[ii] = kmeans2(
            propagated_points[ii], num_clusters, minit='points', missing='raise'
        )

    # Get the statistics
    print('Getting statistics.')
    statistics_from_propagation = _collect_statistics(
        cluster_centers,
        cluster_labels,
        propagated_information,
        extra_information,
        inner,
        outer,
        num_clusters,
    )

    return (
        cluster_centers,
        cluster_labels,
        propagated_points,
        statistics_from_propagation,
    )


def _get_surfaces(radiuses, inner, outer):
    inner_surfaces = np.linspace(radiuses[1], radiuses[0], inner + 2)
    outer_surfaces = np.linspace(radiuses[2], radiuses[1], outer + 2)
    surfaces = np.concatenate((outer_surfaces, inner_surfaces[1:]))
    return surfaces


def _propagate_and_get_info(
    target,
    surfaces,
    propagated_points,
    propagated_information,
    extra_information,
    index,
    num_points,
    time_step,
):
    assert (
        propagated_points[index].shape[0] == num_points
    ), 'Number of points not right.'
    boundary_radiuses = np.array([surfaces[index + 1], surfaces[index - 1]])
    with tqdm() as pbar:
        batch_size = 500
        while True:
            flag = False
            random_indices = np.random.randint(0, num_points, size=(batch_size,))
            initial_locations = propagated_points[index][random_indices]
            ii = 0
            for initial_location in initial_locations:
                previous_location, current_location, target_flag = nestimate.advance_within_concentric_spheres(
                    initial_location, target, boundary_radiuses, time_step, 1
                )
                if target_flag:
                    indicator = 1
                else:
                    indicator = -1

                final_point = nestimate._interpolate(
                    previous_location,
                    current_location,
                    target.center,
                    surfaces[index + indicator],
                )
                if len(propagated_points[index + indicator]) == num_points:
                    extra_temp = np.concatenate(
                        (
                            np.array([index, random_indices[ii], index + indicator]),
                            final_point,
                        )
                    )
                    extra_information.append(extra_temp)
                else:
                    propagated_points[index + indicator].append(final_point)
                    index_temp = len(propagated_points[index + indicator]) - 1
                    propagated_information.append(
                        np.array(
                            [index, random_indices[ii], index + indicator, index_temp],
                            dtype=int,
                        )
                    )

                pbar.update()
                ii += 1
                if (
                    len(propagated_points[index + 1]) == num_points
                    and len(propagated_points[index - 1]) == num_points
                ):
                    propagated_points[index + 1] = np.array(
                        propagated_points[index + 1]
                    )
                    propagated_points[index - 1] = np.array(
                        propagated_points[index - 1]
                    )
                    flag = True
                    break

            if flag:
                break


def _collect_statistics(
    cluster_centers,
    cluster_labels,
    propagated_information,
    extra_information,
    inner,
    outer,
    num_clusters,
):
    num_surfaces = inner + outer + 3
    statistics_from_propagation = [
        [[] for _ in range(num_clusters)] for _ in range(num_surfaces)
    ]
    _process_propagated_info(
        cluster_labels, statistics_from_propagation, propagated_information
    )
    _process_extra_info(
        cluster_centers, cluster_labels, statistics_from_propagation, extra_information
    )
    return statistics_from_propagation


def _process_extra_info(
    cluster_centers, cluster_labels, statistics_from_propagation, extra_information
):
    for info in extra_information:
        centers = cluster_centers[int(info[2])]
        point = info[3:]
        info_temp = info[:3].astype(int)
        index = _assign_clusters(point, centers)
        statistics_from_propagation[info_temp[0]][
            cluster_labels[info_temp[0]][info_temp[1]]
        ].append((info_temp[2], index))


def _assign_clusters(point, centers):
    distances = np.linalg.norm(point - centers, ord=2, axis=1)
    index = np.argmin(distances)
    return index


def _process_propagated_info(
    cluster_labels, statistics_from_propagation, propagated_information
):
    for info in propagated_information:
        statistics_from_propagation[info[0]][cluster_labels[info[0]][info[1]]].append(
            (info[2], cluster_labels[info[2]][info[3]])
        )


def _additional_simulations_for_transition_probabilities(
    target,
    radiuses,
    cluster_centers,
    cluster_labels,
    propagated_points,
    statistics_from_propagation,
    inner,
    outer,
    num_clusters,
    num_trials,
    time_step,
    use_parallel,
    n_split,
):
    surfaces = _get_surfaces(radiuses, inner, outer)
    num_surfaces = len(surfaces)
    if use_parallel:
        manager = mp.Manager()
        statistics_from_propagation = [
            [manager.list(level3) for level3 in level2]
            for level2 in statistics_from_propagation
        ]

    print('Doing additional simulations.')
    # Do more simulations and update statistics_from_propagation
    for ii in range(1, num_surfaces - 1):
        for jj in range(num_clusters):
            print('Doing simulations for surface {}, cluster {}.'.format(ii, jj))
            _do_additional_simulations(
                target,
                radiuses,
                ii,
                jj,
                cluster_centers,
                cluster_labels,
                propagated_points,
                statistics_from_propagation,
                inner,
                outer,
                num_trials,
                time_step,
                use_parallel,
                n_split,
            )

    if use_parallel:
        for ii in range(len(statistics_from_propagation)):
            statistics_from_propagation[ii] = [
                list(level3) for level3 in statistics_from_propagation[ii]
            ]

    # Use statistics_from_propagation to calculate forward and backward probabilities
    forward_probabilities, backward_probabilities = _process_statistics_from_propagation(
        statistics_from_propagation, num_clusters
    )
    return forward_probabilities, backward_probabilities


def _do_additional_simulations(
    target,
    radiuses,
    surface_index,
    cluster_index,
    cluster_centers,
    cluster_labels,
    propagated_points,
    statistics_from_propagation,
    inner,
    outer,
    num_trials,
    time_step,
    use_parallel,
    n_split,
):
    surfaces = _get_surfaces(radiuses, inner, outer)
    cluster_points_indices = np.flatnonzero(
        cluster_labels[surface_index] == cluster_index
    )
    cluster_size = cluster_points_indices.size
    random_indices = np.random.randint(0, cluster_size, size=(num_trials,))
    initial_locations = propagated_points[surface_index][
        cluster_points_indices[random_indices]
    ]
    boundary_radiuses = np.array(
        [surfaces[surface_index + 1], surfaces[surface_index - 1]]
    )
    if use_parallel:
        n_locations = initial_locations.shape[0]

        def worker(indices, q):
            for index in indices:
                initial_location = initial_locations[index]
                previous_location, current_location, target_flag = nestimate.advance_within_concentric_spheres(
                    initial_location, target, boundary_radiuses, time_step, 1
                )
                if target_flag:
                    indicator = 1
                else:
                    indicator = -1

                final_point = nestimate._interpolate(
                    previous_location,
                    current_location,
                    target.center,
                    surfaces[surface_index + indicator],
                )
                centers = cluster_centers[surface_index + indicator]
                index = _assign_clusters(final_point, centers)
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
        for indices in kfold_split(n_locations, n_split):
            process = mp.Process(target=worker, args=(indices, q))
            process_list.append(process)
            process.start()

        for process in process_list:
            process.join()

        q.put(None)
        listener_process.join()
    else:
        for initial_location in tqdm(initial_locations):
            previous_location, current_location, target_flag = nestimate.advance_within_concentric_spheres(
                initial_location, target, boundary_radiuses, time_step, 1
            )
            if target_flag:
                indicator = 1
            else:
                indicator = -1

            final_point = nestimate._interpolate(
                previous_location,
                current_location,
                target.center,
                surfaces[surface_index + indicator],
            )
            centers = cluster_centers[surface_index + indicator]
            index = _assign_clusters(final_point, centers)
            statistics_from_propagation[surface_index][cluster_index].append(
                (surface_index + indicator, index)
            )


def _process_statistics_from_propagation(statistics_from_propagation, num_clusters):
    num_surfaces = len(statistics_from_propagation)
    forward_probabilities = [
        np.zeros((num_clusters, num_clusters), dtype=float)
        for _ in range(num_surfaces - 1)
    ]
    backward_probabilities = [
        np.zeros((num_clusters, num_clusters), dtype=float)
        for _ in range(num_surfaces - 1)
    ]
    for ii in range(1, num_surfaces - 1):
        for jj in range(num_clusters):
            statistics_temp = np.array(statistics_from_propagation[ii][jj])
            forward_transitions = statistics_temp[statistics_temp[:, 0] == ii + 1, 1]
            backward_transitions = statistics_temp[statistics_temp[:, 0] == ii - 1, 1]
            forward_frequencies = np.bincount(
                forward_transitions, minlength=num_clusters
            )
            backward_frequencies = np.bincount(
                backward_transitions, minlength=num_clusters
            )
            assert (
                forward_frequencies.size == num_clusters
                and backward_frequencies.size == num_clusters
            ), "Frequencies not right. ff : {}, bf: {}, nc: {}, ft: {}, bt: {}".format(
                forward_frequencies.size,
                backward_frequencies.size,
                num_clusters,
                forward_transitions,
                backward_transitions,
            )
            total_transitions = float(
                np.sum(forward_frequencies) + np.sum(backward_frequencies)
            )
            assert (
                total_transitions == statistics_temp.shape[0]
            ), '#transitions: {}, forward_frequencies: {}, backward_frequencies: {}, sum: {}'.format(
                total_transitions, forward_frequencies, backward_frequencies
            )
            forward_frequencies = forward_frequencies.astype(float)
            backward_frequencies = backward_frequencies.astype(float)
            forward_probabilities[ii][jj, :] = forward_frequencies / total_transitions
            backward_probabilities[ii][jj, :] = backward_frequencies / total_transitions

    return forward_probabilities, backward_probabilities


def _get_data_driven_binning_hitting_probability(
    forward_probabilities, backward_probabilities, inner, outer, num_clusters
):
    num_surfaces = inner + outer + 3
    middle_index = outer + 1
    un = np.ones((num_clusters,))
    u0 = np.zeros((num_clusters,))
    Q_matrices = [[] for _ in range(num_surfaces)]
    for jj in range(1, num_surfaces - 1):
        try:
            inverse_forward = np.linalg.inv(forward_probabilities[jj])
        except Exception:
            epsilon = 1e-3 * np.min(
                forward_probabilities[jj][forward_probabilities[jj] > 0]
            )
            temp_sum = np.sum(forward_probabilities[jj], axis=1, keepdims=True)
            forward_probabilities[jj] = forward_probabilities[jj] + epsilon * np.eye(
                num_clusters
            )
            forward_probabilities[jj] = (
                temp_sum
                * forward_probabilities[jj]
                / np.sum(forward_probabilities[jj], axis=1, keepdims=True)
            )
            assert np.alltrue(np.sum(forward_probabilities[jj], axis=1) == temp_sum)
            inverse_forward = np.linalg.inv(forward_probabilities[jj])

        matrix_A = inverse_forward
        matrix_B = -np.dot(inverse_forward, backward_probabilities[jj])
        matrix_C = np.eye(num_clusters)
        matrix_D = np.zeros((num_clusters, num_clusters))
        matrix_temp = np.concatenate(
            (
                np.concatenate((matrix_A, matrix_B), axis=1),
                np.concatenate((matrix_C, matrix_D), axis=1),
            ),
            axis=0,
        )
        Q_matrices[jj] = matrix_temp

    Q_product = np.eye(2 * num_clusters)
    for jj in range(1, num_surfaces - 1):
        Q_product = np.dot(Q_matrices[jj], Q_product)
        if jj == middle_index - 1:
            Q_middle = Q_product

    u1 = np.linalg.solve(
        Q_product[:num_clusters, :num_clusters],
        un - np.dot(Q_product[:num_clusters, num_clusters:], u0),
    )

    temp = np.dot(Q_middle, np.concatenate((u1, u0)))
    probability = temp[:num_clusters]
    return probability


def kfold_split(n_locations, n_fold):
    fold_length = int(np.floor(n_locations / n_fold))
    folds = [
        range(ii * fold_length, (ii + 1) * fold_length) for ii in range(n_fold - 1)
    ]
    folds.append(range((n_fold - 1) * fold_length, n_locations))
    return folds
