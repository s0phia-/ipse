import os
import json
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt


def create_run_directories(id):
    time_id = datetime.now().strftime('%Y_%m_%d_%H_%M')
    run_id = time_id + id
    run_id_path = os.path.join("output", run_id)
    print(f"This is the run_id_path {run_id_path}.")
    if not os.path.exists(run_id_path):
        os.makedirs(run_id_path)
    # model_save_name = os.path.join(dir_path, "model.pt")

    models_path = os.path.join(run_id_path, "models")
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    results_path = os.path.join(run_id_path, "results")
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    plots_path = os.path.join(run_id_path, "plots")
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    return run_id_path, models_path, results_path, plots_path


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def process_parameters(param_dict, run_id_path):
    # param_dict["num_tests"] = len(param_dict["test_points"])
    param_dict["plots_path"] = os.path.join(run_id_path, "plots")

    with open(os.path.join(run_id_path, "param_dict.json"), "w") as write_file:
        json.dump(param_dict, write_file, indent=4)

    with open(os.path.join(run_id_path, "param_dict.json"), "r") as read_file:
        param_dict = json.load(read_file)

    p = Bunch(param_dict)
    return p


def plot_learning_curve(plots_path, res_array, x_axis=None, suffix=""):
    if x_axis is None:
        x_axis = np.arange(res_array.shape[1])

    mean_array = np.mean(res_array, axis=0)
    # median_array = np.median(res_array, axis=(0, 2))
    # max_array = np.max(res_array, axis=(0, 2))
    serr_array = np.std(res_array, axis=0) / np.sqrt(res_array.shape[0])

    # Plot and save learning curves.
    fig1, ax1 = plt.subplots()
    ax1.plot(x_axis, mean_array, label="mean")
    # ax1.plot(x_axis, median_array, label="median")
    ax1.fill_between(x_axis, mean_array - serr_array, mean_array + serr_array, alpha=0.2)
    plt.legend()
    fig1.show()
    fig1.savefig(os.path.join(plots_path, "mean_performance" + suffix))
    plt.close()


def plot_multiple_learning_curves(plots_path, compare_results, compare_ids, x_axis, title=""):
    if x_axis is None:
        x_axis = np.arange(compare_results[0].shape[1])

    # Plot and save learning curves.
    fig1, ax1 = plt.subplots()
    for test_results_ix in range(len(compare_results)):
        test_results = compare_results[test_results_ix]
        mean_array = np.mean(test_results, axis=0)
        serr_array = np.std(test_results, axis=0) / np.sqrt(compare_results[0].shape[0])
        ax1.plot(x_axis, mean_array, label=compare_ids[test_results_ix])
        ax1.fill_between(x_axis, mean_array - serr_array, mean_array + serr_array, alpha=0.2)

    plt.title(f'Mean performance{title}')
    plt.xlabel('Iteration')
    plt.ylabel('Mean score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
    fig1.savefig(os.path.join(plots_path, "mean_performance"), bbox_inches='tight')
    plt.close()

    fig1, ax1 = plt.subplots()
    for test_results_ix in range(len(compare_results)):
        test_results = compare_results[test_results_ix]
        mean_array = np.median(test_results, axis=0)
        ax1.plot(x_axis, mean_array, label=compare_ids[test_results_ix])

    plt.title(f'Median performance{title}')
    plt.xlabel('Iteration')
    plt.ylabel('Median score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
    fig1.savefig(os.path.join(plots_path, "median_performance"), bbox_inches='tight')
    plt.close()

    fig1, ax1 = plt.subplots()
    for test_results_ix in range(len(compare_results)):
        test_results = compare_results[test_results_ix]
        mean_array = np.max(test_results, axis=0)
        ax1.plot(x_axis, mean_array, label=compare_ids[test_results_ix])

    plt.title(f'Max performance{title}')
    plt.xlabel('Iteration')
    plt.ylabel('Max score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
    fig1.savefig(os.path.join(plots_path, "max_performance"), bbox_inches='tight')
    plt.close()
