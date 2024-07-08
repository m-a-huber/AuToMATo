# Code to recreate results of experiments
# Uncomment lines 10, 88-90, 101 and 113 to include TTK-clustering algorithm

import os
import json
from functools import partial
from sklearn.base import clone
from sklearn.cluster import AgglomerativeClustering, DBSCAN, HDBSCAN
from eval.finch_subclassed.finch_subclassed import FINCHSubclassed
# from eval.ttk_subclassed.ttk_subclassed import TTKSubclassed
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import fowlkes_mallows_score
import numpy as np
import pandas as pd
from automato import Automato
from persistence_plotting import cs_wong
import clustbench as cb
from warnings import simplefilter
import plotly.graph_objects as gobj

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
pd.options.plotting.backend = "plotly"

data_path = "../clustering-data-v1/"  # data path for clustbench
verbose = True

batch_name = "benchmarks_without_noise"
json_file = f"./eval/{batch_name}.json"
collapse = False  # whether or not to collapse results across ground truths

seed = 42  # seed for random states for Automato
n = 10  # number of Automato iterations
random_states = np.random.RandomState(seed).choice(100, n, replace=False)
epsilons = np.linspace(0, 1, 21, dtype="float")[1:]  # [0.05, 0.1, ...,0.95, 1]

# Algorithms for comparison
clusterers = (
    [
        Automato(
            random_state=random_state,
        )
        for random_state in random_states
    ]
    + [
        AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=epsilon,
            linkage="ward"
        )
        for epsilon in epsilons
    ]
    + [
        AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=epsilon,
            linkage="complete"
        )
        for epsilon in epsilons
    ]
    + [
        AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=epsilon,
            linkage="average"
        )
        for epsilon in epsilons
    ]
    + [
        AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=epsilon,
            linkage="single"
        )
        for epsilon in epsilons
    ]
    + [
        DBSCAN(
            eps=epsilon
        )
        for epsilon in epsilons
    ]
    + [
        HDBSCAN()
    ]
    + [
        FINCHSubclassed()
    ]
    # + [
    #     TTKSubclassed()
    # ]
)
clusterer_names = (
    [f"automato_random_state_{random_state}" for random_state in random_states]
    + [f"linkage_ward_eps_{np.around(epsilon, 2)}" for epsilon in epsilons]
    + [f"linkage_complete_eps_{np.around(epsilon, 2)}" for epsilon in epsilons]
    + [f"linkage_average_eps_{np.around(epsilon, 2)}" for epsilon in epsilons]
    + [f"linkage_single_eps_{np.around(epsilon, 2)}" for epsilon in epsilons]
    + [f"dbscan_eps_{np.around(epsilon, 2)}" for epsilon in epsilons]
    + ["hdbscan"]
    + ["finch"]
    # + ["ttk"]
)
clusterer_shortnames = (
    [f"automato_random_state_{random_state}" for random_state in random_states]
    + [
        "linkage_ward",
        "linkage_complete",
        "linkage_average",
        "linkage_single",
        "dbscan",
        "hdbscan",
        "finch",
        # "ttk"
    ]
)

# Metrics to compute
metrics = [
    fowlkes_mallows_score
]
metric_names = [
    "fms"
]
collapse_str = "_collapsed" if collapse else ""


def vprint(s):
    if verbose:
        print(s)
    else:
        pass


# Generator iterating over benchmarking instances
def get_generator(json_file, with_name=True):
    with open(json_file) as file:
        dictionary = json.load(file)
    for battery in dictionary:
        for benchmark, ixs in dictionary[battery].items():
            X = cb.load_dataset(battery, benchmark, path=data_path).data
            ys = cb.load_dataset(battery, benchmark, path=data_path).labels
            for ix in ixs:
                y = ys[ix] - 1
                bm_name = f"{battery}_{benchmark}_{ix}"
                if with_name:
                    yield X, y, bm_name
                else:
                    yield X, y


# Fit and evaulate algorithms
def fit_eval_clusterers(json_file, clusterers, metrics):
    if not os.path.isdir("./eval/eval_results"):
        os.mkdir("./eval/eval_results")
    # Fit clusterers
    filename = (
        "./eval/eval_results/"
        + f"clusterers_fitted_with_groundtruths_{batch_name}.pkl"
    )
    if os.path.exists(filename):
        df_clusterers_fitted_with_groundtruths = pd.read_pickle(filename)
    else:
        df_clusterers_fitted_with_groundtruths = pd.DataFrame(
            data={},
            index=[
                bm_name
                for X, y, bm_name in get_generator(json_file, with_name=True)
            ],
            dtype=object
        )
    for clusterer, clusterer_name in zip(clusterers, clusterer_names):
        if clusterer_name in df_clusterers_fitted_with_groundtruths.columns:
            vprint(f"Found fitted {clusterer_name} on disk, not fitting.")
        else:
            vprint(f"Fitting {clusterer_name}...")
            df_clusterers_fitted_with_groundtruths[clusterer_name] = [
                (clone(clusterer).fit(minmax_scale(X)), y)
                for X, y, bm_name in get_generator(json_file, with_name=True)
            ]
            vprint(f"Done fitting {clusterer_name}!")
    df_clusterers_fitted_with_groundtruths.to_pickle(filename)
    vprint(f"Saved fitted clusterers with groundtruths to {filename}!")

    def get_first(t): return t[0]
    df_clusterers_fitted = df_clusterers_fitted_with_groundtruths.apply(
        np.vectorize(get_first)
    )
    # Save fitted algorithms to disk
    filename = f"./eval/eval_results/clusterers_fitted_{batch_name}.pkl"
    df_clusterers_fitted.to_pickle(filename)
    vprint(f"Saved fitted clusterers to {filename}!")
    # Compute scores
    for metric, metric_name in zip(metrics, metric_names):
        vprint(f"Computing {metric_name}-scores...")
        df_scores = pd.DataFrame(
            data={},
            index=[
                bm_name
                for X, y, bm_name in get_generator(
                    json_file,
                    with_name=True
                )
            ],
            dtype=object
        )
        for clusterer, clusterer_name in zip(clusterers, clusterer_names):
            vprint(
                f"Computing {metric_name}-score for {clusterer_name}..."
            )
            df_scores = df_scores.assign(
                **{
                    clusterer_name:
                    df_clusterers_fitted_with_groundtruths[
                        [clusterer_name]
                    ].apply(
                        np.vectorize(partial(_apply_metric, metric=metric))
                    )
                }
            )
            vprint(
                f"Done computing {metric_name}-score for {clusterer_name}!"
            )
        # Collapse across ground truths if needed
        if collapse:
            df_scores = df_scores.assign(
                tmp=df_scores.index
            )
            df_scores = df_scores.assign(
                tmp=df_scores["tmp"].apply(lambda s: s[:-2])
            )
            df_scores = df_scores.groupby("tmp").max()
            df_scores.index.name = None
        aut_cols = [
            col
            for col in df_scores.columns
            if col.startswith("automato_random_state")
        ]
        df_scores = df_scores.reindex(sorted(df_scores.columns), axis=1)
        df_scores.insert(n, "automato_var", df_scores[aut_cols].var(1, ddof=0))
        df_scores.insert(n, "automato_std", df_scores[aut_cols].std(1))
        df_scores.insert(n, "automato_mean", df_scores[aut_cols].mean(1))
        # Save scores to disk
        filename_out = (
            f"./eval/eval_results/scores_{metric_name}"
            + f"{collapse_str}_{batch_name}.pkl"
        )
        df_scores.to_pickle(filename_out)
        vprint(f"Saved {metric_name}-scores to {filename_out}!")
    return


def _apply_metric(t, metric):
    labels_true = t[1]
    labels_pred = getattr(t[0], "labels_")
    return metric(
        labels_true[labels_pred != -1],
        labels_pred[labels_pred != -1],
    )


# Create pandas dataframes containing summary of scores
def get_summary_df(df_scores, shortname, metric_name):
    vprint(f"Computing {metric_name}-summary for {shortname}...")
    if shortname in [
        "linkage_ward",
        "linkage_complete",
        "linkage_average",
        "linkage_single",
        "dbscan"
    ]:
        filter_cols = [
            col
            for col in df_scores.columns
            if col.startswith(shortname)
        ]
        df_summary_shortname = df_scores[
            ["automato_mean", "automato_std", "automato_var"]
            + filter_cols
        ]
        shortname_max = f"{shortname}_max"
        shortname_min = f"{shortname}_min"
        df_summary_shortname = df_summary_shortname.assign(
            **{shortname_max: np.max(
                df_summary_shortname[filter_cols], axis=1
            )}
        )
        df_summary_shortname = df_summary_shortname.assign(
            **{shortname_min: np.min(
                df_summary_shortname[filter_cols], axis=1
            )}
        )
        df_summary_shortname = df_summary_shortname[
            [
                "automato_mean",
                "automato_std",
                "automato_var",
                shortname_max,
                shortname_min
            ]
        ]
    else:
        df_summary_shortname = df_scores[
            [
                "automato_mean",
                "automato_std",
                "automato_var",
                shortname
            ]
        ]
    df_summary_shortname_sorted = df_summary_shortname
    if shortname in [
        "linkage_ward",
        "linkage_complete",
        "linkage_average",
        "linkage_single",
        "dbscan"
    ]:
        df_summary_shortname_sorted = df_summary_shortname_sorted.assign(
            diff=df_summary_shortname_sorted["automato_mean"]
            - df_summary_shortname_sorted[f"{shortname}_max"]
        )
    else:
        df_summary_shortname_sorted = df_summary_shortname_sorted.assign(
            diff=df_summary_shortname_sorted["automato_mean"]
            - df_summary_shortname_sorted[shortname]
        )
    df_summary_shortname_sorted = df_summary_shortname_sorted.sort_values(
        by=["diff", "automato_mean"],
        axis=0
    )
    df_summary_shortname_sorted = df_summary_shortname_sorted.drop(
        ["diff"],
        axis=1
    )
    vprint(f"Done computing {metric_name}-score-summary for {shortname}!")
    # Save summary to disk
    filename_out = (
        "./eval/eval_results/"
        + f"summary_{metric_name}_{shortname}{collapse_str}_{batch_name}.pkl"
    )
    df_summary_shortname_sorted.to_pickle(filename_out)
    vprint(
        f"Saved {metric_name}-score-summary "
        f"for {shortname} to {filename_out}!"
    )
    return df_summary_shortname_sorted


# Plot summary dataframes
def plot(df, shortname, score):
    df_plt = df.drop(
        ["automato_std", "automato_var"],
        axis=1
    )
    fig = df_plt.plot(
        template="simple_white",
        labels=dict(
            index="benchmark",
            value=f"{score}-score",
            variable="clusterer"
        )
    )
    dashs = [
        "solid",
        "dot",
        "dashdot"
    ]
    for i in range(df_plt.shape[1]):
        i = i % 8
        fig._data_objs[i].line.color = f"rgb{cs_wong.rgbs[i]}"
        fig._data_objs[i].line.dash = dashs[i]
    # Add bands with std deviation
    xs = list(df.index)
    ys_upper = [x + y for x, y in zip(df.automato_mean, df.automato_std)]
    ys_lower = [x - y for x, y in zip(df.automato_mean, df.automato_std)]
    fig.add_trace(
        gobj.Scatter(
            x=xs+xs[::-1],
            y=ys_upper+ys_lower[::-1],
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        )
    )
    fig.update_yaxes(rangemode="tozero")
    # fig.update_yaxes(range=[0, 1])
    fig.update_layout(
        font_family="monospace",
        width=1200,
        height=450,
    )
    if not os.path.isdir("./eval/pictures"):
        os.mkdir("./eval/pictures")
    # Save plot to disk
    filename_plot = (
        f"./eval/pictures/summary_graph_{metric_name}_"
        + f"{shortname}{collapse_str}_{batch_name}.svg"
    )
    fig.write_image(filename_plot, scale=2)
    vprint(
        f"Saved summary graph for {shortname} "
        f"to {filename_plot}!"
    )
    return fig


if __name__ == "__main__":
    for metric_name in metric_names:
        filename_scores = (
            "./eval/eval_results/"
            + f"scores_{metric_name}{collapse_str}_{batch_name}.pkl"
        )
        if not os.path.exists(filename_scores):
            fit_eval_clusterers(
                json_file=json_file,
                clusterers=clusterers,
                metrics=metrics
            )
        df_scores = pd.read_pickle(
            filename_scores
        )
        for shortname in clusterer_shortnames:
            if not shortname.startswith("automato_random_state"):
                filename_summary = (
                    "./eval/eval_results/"
                    + f"summary_{metric_name}_{shortname}"
                    + f"{collapse_str}_{batch_name}.pkl"
                )
                if not os.path.exists(filename_summary):
                    get_summary_df(
                        df_scores,
                        shortname,
                        metric_name
                    )
                df_summary_shortname = pd.read_pickle(filename_summary)
                if not os.path.isdir("./eval/pictures"):
                    os.mkdir("./eval/pictures")
                filename_plot = (
                    f"./eval/pictures/summary_graph_{metric_name}_"
                    + f"{shortname}{collapse_str}_{batch_name}.svg"
                )
                if not os.path.exists(filename_plot):
                    plot(df_summary_shortname, shortname, metric_name)
