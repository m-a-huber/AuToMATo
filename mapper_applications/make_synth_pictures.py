# Code to recreate results of Mapper applied to concentric circles

import os

import gtda.mapper as mpr
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.datasets import make_circles

from automato import Automato
from dataset_utils import plotting

# Create concentric circles
X, y = make_circles(n_samples=2000, noise=0.05, factor=0.3, random_state=42)
fig = plotting.plot_point_cloud(
    X,
    labels=y,
    to_scale=True
)
fig.update_xaxes(
        title="x",
        tickmode='linear'
)
fig.update_yaxes(
        title="y",
        tickmode='linear'
)
fig.update_layout(
    autosize=False,
    width=400,
    height=400,
)
if not os.path.exists("./mapper_applications/figures/"):
    os.mkdir("./mapper_applications/figures/")
filename = "./mapper_applications/figures/concentric_circles.svg"
fig.write_image(filename)

# Instantiate Mapper
filter_func = mpr.Projection(columns=[0])  # Specify filter
n_intervals = 15  # Specify numbers of intervals to use
overlap_frac = 0.3  # Specify fractional overlap
cover = mpr.CubicalCover(
    n_intervals=n_intervals,
    overlap_frac=overlap_frac
)
n_jobs = -1
clusterers = [
    Automato(random_state=42),
    DBSCAN(),
    HDBSCAN()
]
# Clustering algorithms to be used in Mapper
clusterer_names = [
    "automato",
    "dbscan",
    "hdbscan"
]
for clusterer, clusterer_name in zip(clusterers, clusterer_names):
    pipe = mpr.make_mapper_pipeline(
        filter_func=filter_func,
        cover=cover,
        clusterer=clusterer,
        verbose=False,
        n_jobs=n_jobs,
        min_intersection=1
    )
    # Create Mapper graph
    mapper_graph = pipe.fit_transform(X)
    # Create Mapper figure
    plotly_params = {
        "node_trace": {
            "marker_colorscale": "viridis",
            "marker_showscale": False
        }
    }
    fig = mpr.plot_static_mapper_graph(
        pipe,
        X,
        color_data=y,
        layout_dim=2,
        plotly_params=plotly_params
    )
    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True
    )
    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True
    )
    fig.update_layout(
        autosize=False,
        width=400,
        height=400,
    )
    # Save Mapper figure to disk
    if not os.path.exists("./mapper_applications/figures/"):
        os.mkdir("./mapper_applications/figures/")
    filename = (
        "./mapper_applications/figures/mapper_concentric_circles_"
        + f"{clusterer_name}_{n_intervals}_intervals_"
        + f"{overlap_frac}_overlap.svg"
    )
    fig.write_image(filename)
