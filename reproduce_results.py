import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import subprocess


ROOT_FOLDER = os.path.dirname(os.path.abspath(__file__))


def simple_hitting_prob_test(output_identifier):
    output_folder = '{}/output/{}'.format(ROOT_FOLDER, output_identifier)
    results = {}
    results_fname = '{}/hitting_prob.hdf5'.format(output_folder)
    fig_fname = '{}/figs/simple_hitting_prob_hist.png'.format(ROOT_FOLDER)
    if not os.path.exists(os.path.dirname(fig_fname)):
        os.mkdir(os.path.dirname(fig_fname))

    with h5py.File(results_fname, 'r') as f:
        results['expected_prob'] = f['expected_condenser_hitting_prob'].value
        PA_hitting_prob_list = f['hitting_prob_list'].value[:, 0]

    results['mean_hitting_prob'] = np.mean(PA_hitting_prob_list)
    results['std_hitting_prob'] = np.std(PA_hitting_prob_list)
    results['min'] = np.min(PA_hitting_prob_list)
    results['max'] = np.max(PA_hitting_prob_list)
    results['epsilon'] = np.max(PA_hitting_prob_list) - np.min(PA_hitting_prob_list)
    results['fig_fname'] = fig_fname
    fig, ax = plt.subplots(1, 1)
    ax.hist(PA_hitting_prob_list, bins=7)
    ax.axvline(x=results['expected_prob'][0], linewidth=1, color='r')
    ax.annotate(
        'CHop Probability {:.4f}'.format(results['expected_prob'][0]),
        xy=(results['expected_prob'][0], 25),
        xytext=(results['expected_prob'][0] + 0.1, 27),
        arrowprops=dict(facecolor='black', shrink=0.05),
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel('Hitting Probability')
    ax.set_title(
        'Mean {:.4f}, standard deviation {:.4f}'.format(
            results['mean_hitting_prob'], results['std_hitting_prob']
        )
    )
    fig.savefig(fig_fname, dpi=400)
    return results


def capacity_estimation_test(output_identifier):
    output_folder = '{}/output/{}'.format(ROOT_FOLDER, output_identifier)
    results = {}
    results_fname = '{}/results.hdf5'.format(output_folder)
    with h5py.File(results_fname, 'r') as f:
        results['expected_capacity'] = f['expected_capacity'].value
        results['estimated_capacity'] = f['estimated_capacity'].value

    return results


def nontrivial_hitting_prob_test(output_identifier, CHop_probability):
    output_folder = '{}/output/{}'.format(ROOT_FOLDER, output_identifier)
    results = {}
    results_fname = '{}/hitting_prob.hdf5'.format(output_folder)
    fig_fname = '{}/figs/nontrivial_hitting_prob_hist.png'.format(ROOT_FOLDER)
    if not os.path.exists(os.path.dirname(fig_fname)):
        os.mkdir(os.path.dirname(fig_fname))

    with h5py.File(results_fname, 'r') as f:
        PA_hitting_prob_list = f['hitting_prob_list'].value[:, 0]
        results['time_taken'] = f['time_taken'].value
        results['n_cpus'] = f['n_cpus'].value

    results['mean_hitting_prob'] = np.mean(PA_hitting_prob_list)
    results['std_hitting_prob'] = np.std(PA_hitting_prob_list)
    results['min'] = np.min(PA_hitting_prob_list)
    results['max'] = np.max(PA_hitting_prob_list)
    results['epsilon'] = np.max(PA_hitting_prob_list) - np.min(PA_hitting_prob_list)
    results['mean_time_taken'] = np.mean(results['time_taken'])
    results['std_time_taken'] = np.std(results['time_taken'])
    results['fig_fname'] = fig_fname
    fig, ax = plt.subplots(1, 1)
    ax.hist(PA_hitting_prob_list, bins=7)
    ax.axvline(x=CHop_probability, linewidth=1, color='r')
    ax.annotate(
        'CHop probability {:.4f}'.format(CHop_probability),
        xy=(CHop_probability, 25),
        xytext=(CHop_probability - 0.5, 27),
        arrowprops=dict(facecolor='black', shrink=0.05),
    )

    ax.set_xlim(0, 1)
    ax.set_xlabel('Hitting Probability')
    ax.set_title(
        'Mean {:.4f}, standard deviation {:.4f}'.format(
            results['mean_hitting_prob'], results['std_hitting_prob']
        )
    )
    fig.savefig(fig_fname, dpi=400)
    return results


def hitting_prob_estimation_capacity(output_identifier):
    output_folder = '{}/output/{}'.format(ROOT_FOLDER, output_identifier)
    results = {}
    results_fname = '{}/results.hdf5'.format(output_folder)
    with h5py.File(results_fname, 'r') as f:
        results['hitting_prob'] = f['hitting_prob'].value[0]
        results['time_taken'] = f['time_taken'].value

    return results


if __name__ == '__main__':
    subprocess.call(['pweave', 'report.mdw'])
    subprocess.call(['pandoc', '-t', 'latex', '-o', 'report.pdf', 'report.md'])
