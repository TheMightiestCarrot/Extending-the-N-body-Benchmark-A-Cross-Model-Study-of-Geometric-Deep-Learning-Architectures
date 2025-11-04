import argparse
import glob
import json
import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import entropy, kstest


def load_data(file_path):
    # Replace with actual data loading logic
    return np.load(file_path)


def js_divergence(p, q, epsilon=1000):
    """
    Calculate the Jensen-Shannon Divergence between two probability distributions,
    adding a small epsilon to avoid division by zero.

    Args:
    p (np.array): Probability distribution 1.
    q (np.array): Probability distribution 2.
    epsilon (float): Small value to add to distributions to prevent division by zero.

    Returns:
    float: Jensen-Shannon Divergence.
    """
    p = np.asarray(p)
    q = np.asarray(q)
    p += epsilon
    q += epsilon
    p /= p.sum()
    q /= q.sum()

    m = 0.5 * (p + q)
    jsd = 0.5 * (entropy(p, m) + entropy(q, m))
    return jsd


def kl_divergence(p, q, epsilon=1e-10):
    """
    Calculate the Kullback-Leibler Divergence between two probability distributions,
    adding a small epsilon to avoid division by zero in log calculations.

    Args:
    p (np.array): Probability distribution 1.
    q (np.array): Probability distribution 2.
    epsilon (float): Small value to add to distributions to prevent division by zero in log.

    Returns:
    float: Kullback-Leibler Divergence.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p += epsilon
    q += epsilon
    p /= p.sum()
    q /= q.sum()

    kl_div = np.sum(p * np.log(p / q))
    return kl_div


def plot_kl_divergence_over_time(data_collections, data_name, save_dir):
    batch_values = []
    div_values = []

    for batch, distributions in data_collections.items():
        if distributions is None:
            continue
        ground_truth = distributions["ground truth"]
        predicted = distributions["predicted"]

        if ground_truth and predicted:
            div = kl_divergence(ground_truth[data_name], predicted[data_name])
            batch_values.append(int(batch))
            div_values.append(div)

    sorted_indices = np.argsort(batch_values)
    batch_values = np.array(batch_values)[sorted_indices]
    div_values = np.array(div_values)[sorted_indices]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=batch_values, y=div_values, mode='lines+markers', name=f"KL Divergence {data_name}"))
    fig.update_layout(title="KL Divergence for Different Training Batches",
                      xaxis_title="Training Batch",
                      yaxis_title="KL Divergence")
    fig.update_yaxes(fixedrange=False)

    # Add JavaScript to dynamically update y-axis range
    fig.write_html(f"{save_dir}/kl_divergence_{data_name}.html", include_plotlyjs='cdn', full_html=False)
    with open(f"{save_dir}/kl_divergence_{data_name}.html", 'a') as f:
        f.write('''
<script>
document.addEventListener('DOMContentLoaded', function() {
    var graphDiv = document.getElementById('kl-divergence-graph');
    graphDiv.on('plotly_relayout', function(eventdata) {
        if (eventdata['xaxis.range[0]'] && eventdata['xaxis.range[1]']) {
            var x0 = eventdata['xaxis.range[0]'];
            var x1 = eventdata['xaxis.range[1]'];
            var batch_values = ''' + str(batch_values.tolist()) + ''';
            var div_values = ''' + str(div_values.tolist()) + ''';
            var mask = batch_values.map(function(value) {
                return value >= x0 && value <= x1;
            });
            var filtered_div_values = div_values.filter(function(value, index) {
                return mask[index];
            });
            if (filtered_div_values.length > 0) {
                var y0 = Math.min.apply(null, filtered_div_values);
                var y1 = Math.max.apply(null, filtered_div_values);
                Plotly.relayout(graphDiv, 'yaxis.range', [y0, y1]);
            }
        }
    });
});
</script>
''')

def plot_energy_conservation_over_time(energy_statistics, save_dir):
    energy_values_predicted = []
    batch_values = []

    for batch, energy_statistic in energy_statistics.items():
        if energy_statistic is None:
            continue
        energy_values_predicted.append(
            energy_statistic["predicted"]["data"][2]["mean"][0]
            - (
                sum(energy_statistic["predicted"]["data"][2]["mean"])
                / len(energy_statistic["predicted"]["data"][2]["mean"])
            )
        )
        batch_values.append(int(batch))

    sorted_indices = np.argsort(batch_values)
    batch_values = np.array(batch_values)[sorted_indices]
    energy_values_predicted = np.array(energy_values_predicted)[sorted_indices]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=batch_values, y=energy_values_predicted, mode='lines+markers', name="Energy Conservation Divergence"))
    fig.update_layout(title="Energy Conservation Over Time",
                      xaxis_title="Training Batch",
                      yaxis_title="Energy")
    fig.update_yaxes(fixedrange=False)
    fig.write_html(f"{save_dir}/energy_conservation_over_time.html", include_plotlyjs='cdn', full_html=False)
    with open(f"{save_dir}/energy_conservation_over_time.html", 'a') as f:
        f.write('''
<script>
document.addEventListener('DOMContentLoaded', function() {
    var graphDiv = document.getElementById('kl-divergence-graph');
    graphDiv.on('plotly_relayout', function(eventdata) {
        if (eventdata['xaxis.range[0]'] && eventdata['xaxis.range[1]']) {
            var x0 = eventdata['xaxis.range[0]'];
            var x1 = eventdata['xaxis.range[1]'];
            var batch_values = ''' + str(batch_values.tolist()) + ''';
            var energy_values_predicted = ''' + str(energy_values_predicted.tolist()) + ''';
            var mask = batch_values.map(function(value) {
                return value >= x0 && value <= x1;
            });
            var filtered_div_values = div_values.filter(function(value, index) {
                return mask[index];
            });
            if (filtered_div_values.length > 0) {
                var y0 = Math.min.apply(null, filtered_div_values);
                var y1 = Math.max.apply(null, filtered_div_values);
                Plotly.relayout(graphDiv, 'yaxis.range', [y0, y1]);
            }
        }
    });
});
</script>
''')
        
def plot_difference_in_distributions(data_collections, data_name, save_dir):
    """
    Plots difference between ground truth and infered data in mean/std of distributions of each training batch.
    """
    # Prepare lists to accumulate data points
    batch_values = []
    avg_diff_values = []
    std_diff_values = []

    # Assuming data_collections is a dictionary with keys as 'dt' and values as lists of data entries
    for batch, distributions in data_collections.items():
        if distributions is None:
            continue
        ground_truth = distributions["ground truth"]
        predicted = distributions["predicted"]

        if ground_truth and predicted:
            abs_avg_diff = abs((sum(ground_truth[data_name])/len(ground_truth[data_name])) - 
                               (sum(predicted[data_name])/len(predicted[data_name])))
            abs_std_diff = abs(np.std(ground_truth[data_name]) - np.std(predicted[data_name]))

            # Convert dt to float and accumulate data points
            batch_values.append(int(batch))
            avg_diff_values.append(abs_avg_diff)
            std_diff_values.append(abs_std_diff)

    sorted_indices = np.argsort(batch_values)
    batch_values = np.array(batch_values)[sorted_indices]
    avg_diff_values = np.array(avg_diff_values)[sorted_indices]
    std_diff_values = np.array(std_diff_values)[sorted_indices]

    fig = go.Figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=batch_values, y=avg_diff_values, mode='lines+markers',
                             name=f"diff_abs_mean {data_name}"))
    fig.add_trace(go.Scatter(x=batch_values, y=std_diff_values, mode='lines+markers',
                             name=f"diff_abs_std {data_name}"), secondary_y=True)
    fig.update_layout(title="Difference of Distributions Between Training Batches",
                      xaxis_title="Training Batch")
    fig.update_yaxes(title_text="Difference Between Averages", secondary_y=False)
    fig.update_yaxes(title_text="Difference Between Standard Deviations", secondary_y=True)
    fig.update_yaxes(fixedrange=False)
    fig.write_html(f"{save_dir}/difference_of_distributions_{data_name}.html",
                    include_plotlyjs='cdn', full_html=False)

def plot_ks_statistics(data_collections, data_name, save_dir):
    """
    Performs the two-sample Kolmogorov-Smirnov test for goodness of fit.
    """
     # Prepare lists to accumulate data points
    batch_values = []
    p_values = []
    stats_values = []

    # Assuming data_collections is a dictionary with keys as 'dt' and values as lists of data entries
    for batch, distributions in data_collections.items():
        if distributions is None:
            continue
        ground_truth = distributions["ground truth"]
        predicted = distributions["predicted"]
        if ground_truth and predicted:
            _stats = kstest(ground_truth[data_name], predicted[data_name])

            # Convert dt to float and accumulate data points
            batch_values.append(int(batch))
            p_values.append(_stats[1])
            stats_values.append(_stats[0])

    sorted_indices = np.argsort(batch_values)
    batch_values = np.array(batch_values)[sorted_indices]
    p_values = np.array(p_values)[sorted_indices]

    fig = go.Figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=batch_values, y=p_values, mode='lines+markers',
                             name=f"p_vlaue {data_name}"))
    fig.add_trace(go.Scatter(x=batch_values, y=stats_values, mode='lines+markers',
                             name=f"statistics value {data_name}"), secondary_y=True)
    fig.update_layout(title="P value and statistics value of Kolmogorov-Smirnov test between ground ",
                      xaxis_title="Training Batch")
    fig.update_yaxes(title_text="P values", secondary_y=False)
    fig.update_yaxes(title_text="KS statistics", secondary_y=True)
    fig.write_html(f"{save_dir}/ks_test_{data_name}.html",
                    include_plotlyjs='cdn', full_html=False)

def main():
    parser = argparse.ArgumentParser(description="Plot metamacro data over training batches")
    parser.add_argument("--model-dir", type=str, required=False, help="Directory containing the data files")
    parser.add_argument("--save-dir", type=str, required=False, help="Directory to save the plot")
    args = parser.parse_args()

    checkpoint_folder = f"{args.model_dir}/checkpoints"
    print(f"Checkpoint folder: {checkpoint_folder}")

    sticking_distributions = {}
    collision_distributions = {}
    difference_distributions = {}
    feature_distributions = {}
    energy_statistics = {}
    leaving_distributions = {}
    momentum_distributions = {}
    com_distance_distributions = {}

    def load_data(file_name, checkpoint_path):
        file_path = glob.glob(os.path.join(checkpoint_path, "generated_trajectories", "*", "plots", file_name))
        if file_path:
            with open(file_path[0], "r") as json_file:
                data = json.load(json_file)
                return data
        else:
            print(f"File {file_name} not found in {checkpoint_path}")
            return None

    for checkpoint_dir in os.listdir(checkpoint_folder):
        checkpoint_path = os.path.join(checkpoint_folder, checkpoint_dir)
        if os.path.isdir(checkpoint_path):
            sticking_distributions[checkpoint_dir] = load_data("sticking_distributions.json", checkpoint_path)
            collision_distributions[checkpoint_dir] = load_data("collision_distributions.json", checkpoint_path)
            difference_distributions[checkpoint_dir] = load_data("difference_distributions.json", checkpoint_path)
            feature_distributions[checkpoint_dir] = load_data("feature_distributions.json", checkpoint_path)
            energy_statistics[checkpoint_dir] = load_data("energy_statistics.json", checkpoint_path)
            leaving_distributions[checkpoint_dir] = load_data("leaving_distribution.json", checkpoint_path)
            momentum_distributions[checkpoint_dir] = load_data('momentum_statistics.json', checkpoint_path)
            com_distance_distributions[checkpoint_dir] = load_data('max_com_distance_distribution.json', checkpoint_path)

    save_dir = f"{checkpoint_folder}/metaplots"
    os.makedirs(save_dir, exist_ok=True)
    plot_kl_divergence_over_time(sticking_distributions, "sticking_histogram", save_dir)
    plot_kl_divergence_over_time(collision_distributions, "collision_histogram", save_dir)
    plot_kl_divergence_over_time(difference_distributions, "position_difference", save_dir)
    plot_kl_divergence_over_time(difference_distributions, "velocity_difference", save_dir)
    plot_kl_divergence_over_time(feature_distributions, "position", save_dir)
    plot_kl_divergence_over_time(feature_distributions, "velocity", save_dir)
    plot_difference_in_distributions(sticking_distributions, "sticking_histogram", save_dir)
    plot_difference_in_distributions(collision_distributions, "collision_histogram", save_dir)
    plot_difference_in_distributions(difference_distributions, "position_difference", save_dir)
    plot_difference_in_distributions(difference_distributions, "velocity_difference", save_dir)
    plot_difference_in_distributions(feature_distributions, "position", save_dir)
    plot_difference_in_distributions(feature_distributions, "velocity", save_dir)
    plot_difference_in_distributions(leaving_distributions, "leaving_count", save_dir)
    plot_difference_in_distributions(momentum_distributions, "momentum_statistics", save_dir)
    plot_difference_in_distributions(com_distance_distributions, "com_movement", save_dir)
    plot_ks_statistics(sticking_distributions, "sticking_histogram", save_dir)
    plot_ks_statistics(collision_distributions, "collision_histogram", save_dir)
    plot_ks_statistics(difference_distributions, "position_difference", save_dir)
    plot_ks_statistics(difference_distributions, "velocity_difference", save_dir)
    plot_ks_statistics(feature_distributions, "position", save_dir)
    plot_ks_statistics(feature_distributions, "velocity", save_dir)
    plot_ks_statistics(leaving_distributions, "leaving_count", save_dir)
    plot_ks_statistics(momentum_distributions, "momentum_statistics", save_dir)
    plot_ks_statistics(com_distance_distributions, "com_movement", save_dir)
    plot_energy_conservation_over_time(energy_statistics, save_dir)

    print(f"Plots saved to {save_dir}")


if __name__ == "__main__":
    main()

