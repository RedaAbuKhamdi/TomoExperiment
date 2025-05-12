import sys
import numpy as np

from os import environ, listdir, makedirs, path
from matplotlib import pyplot as plt

from metricsdata import MetricsData
import matplotlib.ticker as mticker
from tqdm import tqdm

from matplotlib import colors as mcolors
import random

def generate_distinct_colors(labels):
    color_names = list(mcolors.TABLEAU_COLORS.values())
    random.shuffle(color_names)
    return dict(zip(labels, color_names[:len(labels)]))

sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import config
def iterate_datasets():
    for experiment_path in config.EVALUATION_PATH.iterdir():
        experiment = experiment_path.as_posix().split("/")[-1]
        for dataset_path in experiment_path.iterdir():
            dataset = dataset_path.as_posix().split("/")[-1]
            for algorithm_path in dataset_path.iterdir():
                algorithm = algorithm_path.as_posix().split("/")[-1]
                dataset_config = {
                    "experiment": experiment,
                    "name": dataset,
                    "algorithm": algorithm,
                    "evaluations": []
                }
                for data in algorithm_path.glob("*.csv"):
                    data_name = data.as_posix().split("/")[-1]
                    dataset_config["evaluations"].append({
                        "type": data_name.split(".")[0].replace("_", " ").capitalize(),
                        "path": data.as_posix()
                    })
                yield dataset_config
def delta_from_values(values : np.ndarray) -> np.ndarray:
    result = np.zeros_like(values)
    for i in range(result.shape[0] - 1):
        result[i] = values[i+1] - values[i]
    return result
def scatter_plots(algorithm_data: dict, algorithm: str, colors: dict):
    """
    For each metric (iou, boundarydice, mse, …), plot a curve
    of metric vs. #angles, one line per dataset.
    """
    datasets      = algorithm_data["datasets"]
    angles        = algorithm_data["angles"].astype(int)
    metric_values = algorithm_data["metric_values"]
    metrics       = list(metric_values[datasets[0]].keys())

    out_dir = config.VISUALIZATION_PATH / algorithm
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric_name in metrics:
        fig, ax = plt.subplots(figsize=(8,5), dpi=120)

        for dataset in datasets:
            y = metric_values[dataset][metric_name]
            ax.plot(angles, y,
                    marker='o', linestyle='-',
                    color=colors[dataset],
                    label=dataset)

        ax.set_xscale('log', base=2)
        ax.set_xticks(angles)
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.tick_params(axis='x', rotation=45)

        ax.grid(which='major', axis='y', linestyle='--', alpha=0.5)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Number of Angles", fontsize=12)
        ax.set_ylabel(metric_name.capitalize(),     fontsize=12)
        ax.set_title(f"{metric_name.capitalize()} vs. Angle", fontsize=14)

        ax.legend(title="Dataset", loc='upper left', bbox_to_anchor=(1.02,1), borderaxespad=0)

        plt.tight_layout()
        plt.savefig(out_dir / f"{metric_name}_curve.png", bbox_inches='tight')
        plt.close(fig)

def scatter_plots_per_dataset(algorithm_data: dict,
                              algorithm: str,
                              ground_truth : bool = False):
    datasets      = algorithm_data["datasets"]
    angles        = algorithm_data["angles"].astype(int)
    metric_values = algorithm_data["metric_values"]
    for dataset in datasets:
        # figure setup
        fig, ax = plt.subplots(figsize=(8,5), dpi=120)

        # determine which metrics we're plotting
        metrics = list(metric_values[dataset].keys())
        cmap    = plt.get_cmap('tab10')
        metric_colors = {m: cmap(i) for i, m in enumerate(metrics)}

        # draw each metric as a line + marker
        for metric in metrics:
            y = metric_values[dataset][metric]

            ax.plot(angles, y,
                    marker='o',
                    linestyle='-',
                    color=metric_colors[metric],
                    label=metric.upper())

        # log-2 x-axis so 8→16→32→… are evenly spaced
        ax.set_xscale('log', base=2)
        ax.set_xticks(angles)
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())

        # grid, labels, legend
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Number of Angles", fontsize=12)
        ax.set_ylabel("Metric Value", fontsize=12)
        ax.set_title(f"{dataset}: Metric vs. Angle {"Neighbor" if not ground_truth else "Ground Truth"}", fontsize=14)
        ax.legend(title="", loc='lower right')

        plt.tight_layout()

        # save
        folder = config.VISUALIZATION_PATH / algorithm / "per_dataset"
        folder.mkdir(parents=True, exist_ok=True)
        out_path = folder / f"{dataset}_{"neighbor" if not ground_truth else "ground_truth"}_metrics.png"
        plt.savefig(out_path)
        plt.close(fig)
def scatter_plots_mean_ground_truth(algorithm_data: dict, algorithm : str, colors : list):
    datasets = algorithm_data["datasets"]
    angles = algorithm_data["angles"]
    metric_values = algorithm_data["metric_values"]
    metrics = metric_values[datasets[0]].keys()

    x = np.arange(len(angles))  
    for metric_name in metrics:
        plt.clf()
        values = np.zeros(angles.shape)
        for i, dataset in enumerate(datasets):
            values += metric_values[dataset][metric_name]
        values /= len(datasets)
        plt.scatter(angles.astype(str), values, color=colors[dataset])

        if delta:
            plt.title("Bubble Chart of mean {} differences values across datasets".format(metric_name))
        else:
            plt.title("Bubble Chart of mean {} values across datasets".format(metric_name))

        plt.xlabel("Number of Angles")
        plt.ylabel("Metric Value")
        plt.tight_layout()
        plt.legend()
        folder = config.VISUALIZATION_PATH / "threshold" / algorithm 
        makedirs(folder, exist_ok=True)
        plt.savefig("{0}/{1}_scatter_plot_mean.png".format(folder, metric_name))

if __name__ == "__main__":
    print("Start visualization")
    metrics_data = MetricsData()
    datasets = metrics_data.get_datasets()
    colors = generate_distinct_colors(datasets)
    for algorithm, data in metrics_data.get_per_algorithm_data("Neighbor metrics"):
        print("Processing algorithm {}".format(algorithm))
        scatter_plots(data, algorithm, colors)
        scatter_plots_per_dataset(data, algorithm)
    for algorithm, data in metrics_data.get_per_algorithm_data("Ground truth metrics"):
        print("Processing algorithm {}".format(algorithm))
        scatter_plots_per_dataset(data, algorithm, True)
    
    for algorithm in metrics_data.get_algorithms():
        print("Processing algorithm {} - thresholds".format(algorithm))
        thresholds = np.arange(0, 1.1, 0.05)
        for metric in ["iou", "boundarydice"]:
            angles = []
            metrics = []
            for threshold in tqdm(thresholds):
                angle, metric_value = metrics_data.get_threshold_data(threshold, algorithm, metric)
                angles.append(angle)
                metrics.append(metric_value)
            angles = np.array(angles)
            metrics = np.array(metrics) 
            angles_gt, mean_metrics_gt = metrics_data.get_mean_ground_truth(algorithm, metric)
            plt.clf()
            fig, ax = plt.subplots(figsize=(8,5), dpi=120)
            ax.plot(angles,        metrics,        marker='o', linestyle='-',  label='Thresholds')
            ax.plot(angles_gt,     mean_metrics_gt,marker='s', linestyle='--', label='GT Mean')
            ax.set_xscale('log', base=2)
            ax.set_xticks(angles_gt)
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            ax.set_xlabel("Number of Angles")
            ax.set_ylabel(f"Mean {metric.title()}")
            ax.set_ylim(0,1)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            ax.grid(axis='x', linestyle='--', alpha=0.5)
            ax.legend()
            plt.tight_layout()
            plt.legend()
            folder = config.VISUALIZATION_PATH / "threshold" / algorithm
            makedirs(folder, exist_ok=True)
            plt.savefig("{0}/{1}_threshold.png".format(folder, metric))