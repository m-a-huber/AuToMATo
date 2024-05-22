# Code to recreate results of experiments
# Uncomment lines 10, 84-86, 97 and 108 to include TTK-clustering algorithm

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

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
pd.options.plotting.backend = "plotly"

data_path = "../clustering-data-v1/"  # data path for clustbench
verbose = True

batch_name = "benchmarks_without_noise"
json_file = f"./eval/{batch_name}.json"
collapse = False  # whether or not to collapse results across ground truths

random_state = 42  # random state for Automato
epsilons = np.linspace(0, 1, 21, dtype="float")[1:]  # [0.05, 0.1, ...,0.95, 1]

# Algorithms for comparison
clusterers = (
    [
        Automato(
            random_state=random_state,
        )
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
    ["automato"]
    + [f"linkage_ward_eps_{np.around(epsilon, 2)}" for epsilon in epsilons]
    + [f"linkage_complete_eps_{np.around(epsilon, 2)}" for epsilon in epsilons]
    + [f"linkage_average_eps_{np.around(epsilon, 2)}" for epsilon in epsilons]
    + [f"linkage_single_eps_{np.around(epsilon, 2)}" for epsilon in epsilons]
    + [f"dbscan_eps_{np.around(epsilon, 2)}" for epsilon in epsilons]
    + ["hdbscan"]
    + ["finch"]
    # + ["ttk"]
)
clusterer_shortnames = [
    "automato",
    "linkage_ward",
    "linkage_complete",
    "linkage_average",
    "linkage_single",
    "dbscan",
    "hdbscan",
    "finch",
    # "ttk"
]

# Metrics to compute
metrics = [
    fowlkes_mallows_score
]
metric_names = [
    "fms"
]


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
            vprint(f"Found fitted {clusterer_name } on disk, not fitting.")
        else:
            vprint(f"Fitting {clusterer_name}...")
            df_clusterers_fitted_with_groundtruths[clusterer_name] = [
                (clone(clusterer).fit(minmax_scale(X)), y)
                for X, y, bm_name in get_generator(json_file, with_name=True)
            ]
            df_clusterers_fitted_with_groundtruths.assign(
                **{clusterer_name: [
                    (clone(clusterer).fit(minmax_scale(X)), y)
                    for X, y, bm_name in get_generator(
                        json_file,
                        with_name=True
                    )
                ]}
            )
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
        filename = f"./eval/eval_results/scores_{metric_name}_{batch_name}.pkl"
        if os.path.exists(filename):
            df_scores = pd.read_pickle(filename)
        else:
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
            if clusterer_name in df_scores.columns:
                vprint(
                    f"Found {metric_name}-scores for {clusterer_name } "
                    "on disk, not computing."
                )
            else:
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
        df_scores.to_pickle(filename)
        # Save scores to disk
        vprint(f"Saved {metric_name}-scores to {filename}!")
    return


def _apply_metric(t, metric):
    labels_true = t[1]
    labels_pred = getattr(t[0], "labels_")
    return metric(
        labels_true[labels_pred != -1],
        labels_pred[labels_pred != -1],
    )


# Create pandas dataframes containing summary of scores
def get_summary_df(df_scores, shortname, metric_name, collapse=False):
    if collapse:
        filename = (
            "./eval/eval_results/"
            + f"summary_{metric_name}_{shortname}_collapsed_{batch_name}.pkl"
        )
    else:
        filename = (
            "./eval/eval_results/"
            + f"summary_{metric_name}_{shortname}_{batch_name}.pkl"
        )
    if os.path.exists(filename):
        vprint(
            f"Found {metric_name}-score-summary for "
            f"{shortname} on disk, not computing."
        )
        df_summary_shortname_sorted = pd.read_pickle(filename)
        return df_summary_shortname_sorted
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
        df_summary_shortname = df_scores[["automato"] + filter_cols]
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
            ["automato", shortname_max, shortname_min]
        ]
    else:
        df_summary_shortname = df_scores[
            ["automato", shortname]
        ]
    if collapse:
        df_summary_shortname = df_summary_shortname.assign(
            tmp=df_summary_shortname.index
        )
        df_summary_shortname = df_summary_shortname.assign(
            tmp=df_summary_shortname["tmp"].apply(lambda s: s[:-2])
        )
        df_summary_shortname = df_summary_shortname.groupby("tmp").max()
        df_summary_shortname.index.name = None
    df_summary_shortname_sorted = df_summary_shortname
    if shortname in [
        "linkage_ward",
        "linkage_complete",
        "linkage_average",
        "linkage_single",
        "dbscan"
    ]:
        df_summary_shortname_sorted = df_summary_shortname_sorted.assign(
            diff=df_summary_shortname_sorted["automato"]
            - df_summary_shortname_sorted[f"{shortname}_max"]
        )
    else:
        df_summary_shortname_sorted = df_summary_shortname_sorted.assign(
            diff=df_summary_shortname_sorted["automato"]
            - df_summary_shortname_sorted[shortname]
        )
    df_summary_shortname_sorted = df_summary_shortname_sorted.sort_values(
        by=["diff", "automato"],
        axis=0
    )
    df_summary_shortname_sorted = df_summary_shortname_sorted.drop(
        ["diff"],
        axis=1
    )
    vprint(f"Done computing {metric_name}-score-summary for {shortname}!")
    df_summary_shortname_sorted.to_pickle(filename)
    vprint(
        f"Saved {metric_name}-score-summary "
        f"for {shortname} to {filename}!"
    )
    return df_summary_shortname_sorted


# Plot summary dataframes
def plot(df, score):
    fig = df.T.plot(
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
    for i in range(df.shape[0]):
        i = i % 8
        fig._data_objs[i].line.color = f"rgb{cs_wong.rgbs[i]}"
        fig._data_objs[i].line.dash = dashs[i]
    fig.update_layout(
        font_family="monospace",
        width=1200,
        height=450,
    )
    return fig


if __name__ == "__main__":
    fit_eval_clusterers(
            json_file=json_file,
            clusterers=clusterers,
            metrics=metrics
        )
    for metric_name in metric_names:
        filename = (
            "./eval/eval_results/"
            + f"scores_{metric_name}_{batch_name}.pkl"
        )
        df_scores = pd.read_pickle(
            filename
        )
        for shortname in clusterer_shortnames[1:]:
            df_summary_shortname = get_summary_df(
                df_scores,
                shortname,
                metric_name,
                collapse=collapse
            )
            fig = plot(df_summary_shortname.T, metric_name)
            if not os.path.isdir("./eval/pictures"):
                os.mkdir("./eval/pictures")
            if collapse:
                filename_plot = (
                    "./eval/pictures/summary_graph_"
                    + f"{metric_name}_{shortname}_collapsed_{batch_name}.svg"
                )
            else:
                filename_plot = (
                    "./eval/pictures/summary_graph_"
                    + f"{metric_name}_{shortname}_{batch_name}.svg"
                )
            fig.write_image(filename_plot, scale=2)
