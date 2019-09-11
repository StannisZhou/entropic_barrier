import numpy as np

import golf_course.estimate.numba as nestimate
from golf_course.core.target import Target
from golf_course.estimate.capacity import estimate_capacity


class ToyModel(object):
    """
    ToyModel
    This class defines the toy model we are going to use.
    For simplicity, in our program, we will simply use 0 for the constant region
    outside all the targets.
    """

    def __init__(self, time_step, target_param_list):
        """__init__
        Parameters
        ----------
        time_step: float
            The time step we are going to use for the simulation
        target_param_list: [dict]
            A list of dict, each of which contains the parameters for a
            particular target
        """
        self.time_step = time_step
        assert _check_compatibility(target_param_list, 1)
        self.target_list = [Target(**params) for params in target_param_list]

    def do_naive_simulation(self, current_location):
        radiuses = np.array([target.radiuses[1] for target in self.target_list])
        centers = np.array([target.center for target in self.target_list])
        distances = np.linalg.norm(
            current_location.reshape((1, -1)) - centers, ord=2, axis=1
        )
        assert all(distances > radiuses)
        while True:
            previous_location, current_location, index = nestimate.advance_flat_regions(
                current_location, centers, radiuses, self.time_step
            )
            target = self.target_list[index]
            boundary_radiuses = np.array([target.radiuses[0], target.radiuses[1]])
            previous_location, current_location, target_flag = nestimate.advance_within_concentric_spheres(
                current_location, target, boundary_radiuses, self.time_step, 1
            )
            if target_flag:
                break

        return index

    def estimate_hitting_prob(self, capacity_estimation_param_list):
        n_targets = len(self.target_list)
        hitting_prob = np.zeros(n_targets)
        for ii in range(n_targets):
            target = self.target_list[ii]
            hitting_prob[ii], _ = estimate_capacity(
                target, **capacity_estimation_param_list[ii]
            )

        hitting_prob = hitting_prob / np.sum(hitting_prob)
        return hitting_prob


def _check_compatibility(target_param_list, boundary_radius):
    n_spheres = len(target_param_list)
    for target_param in target_param_list:
        center = target_param['center']
        radius = target_param['radiuses'][0]
        if np.linalg.norm(center) - radius > boundary_radius:
            return False

    for ii in range(n_spheres - 1):
        for jj in range(ii + 1, n_spheres):
            center1 = target_param_list[ii]['center']
            radius1 = target_param_list[ii]['radiuses'][2]
            center2 = target_param_list[jj]['center']
            radius2 = target_param_list[jj]['radiuses'][2]
            if np.linalg.norm(center1 - center2) < radius1 + radius2:
                return False

    return True
