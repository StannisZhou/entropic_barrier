import numpy as np

from golf_course.core.target import Target
from golf_course.estimate.capacity import estimate_capacity

n_dim = 5
radiuses = np.array([0.02, 0.05, 0.1])
target_params = {
    'center': np.zeros(n_dim),
    'radiuses': radiuses,
    'energy_type': 'flat',
    'energy_params': {},
}
target = Target(**target_params)
capacity_estimation_params = {
    'inner': 1,
    'outer': 1,
    'num_points': int(1e2),
    'num_clusters': 3,
    'num_trials': int(3e3),
    'time_step': 1e-07,
    'use_parallel': False,
    'n_split': 1,
    'n_surfaces_gradients_estimation': 16,
}
capacity = estimate_capacity(target, **capacity_estimation_params)
expected_capacity = (n_dim - 2) / (
    radiuses[0] ** (2 - n_dim) - radiuses[2] ** (2 - n_dim)
)
print('Expected capacity {}'.format(expected_capacity))
print('Estimated capacity {}'.format(capacity))
