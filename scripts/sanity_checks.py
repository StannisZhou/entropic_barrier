import numpy as np

from golf_course.core.target import Target
from golf_course.estimate.capacity import estimate_capacity

target_params = {
    'center': np.zeros(5),
    'radiuses': np.array([0.02, 0.05, 0.1]),
    'energy_type': 'flat',
    'energy_params': {},
}
target = Target(**target_params)
capacity_estimation_params = {
    'inner': 20,
    'outer': 1,
    'num_points': int(1e2),
    'num_clusters': 3,
    'num_trials': int(1e3),
    'time_step': 1e-07,
    'use_parallel': False,
    'n_split': 1,
}
estimate_capacity(target, **capacity_estimation_params)
