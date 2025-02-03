# Code to recreate results of Mapper applied to diabetes data

import os

import gtda.mapper as mpr  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
from sklearn.cluster import DBSCAN, HDBSCAN  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from automato import Automato
from mapper_applications.eccentricity_subclassed import EccentricitySubclassed

# Load diabetes data
df = pd.read_csv("./mapper_applications/chemdiab.csv", index_col=[0])
X, y = df.values[:, :-1], df.values[:, -1]
X = StandardScaler().fit_transform(X)
y = np.where(y == "Normal", 0, y)
y = np.where(y == "Chemical_Diabetic", 1, y)
y = np.where(y == "Overt_Diabetic", 2, y)

# Instantiate Mapper
filter_func = EccentricitySubclassed(exponent=np.inf)  # Specify filter
n_intervals_list = [4]  # Specify numbers of intervals to use
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
overlap_frac = 0.5  # Specify fractional overlap
for n_intervals in n_intervals_list:
    for clusterer, clusterer_name in zip(clusterers, clusterer_names):
        cover = mpr.CubicalCover(
            n_intervals=n_intervals,
            overlap_frac=overlap_frac
        )
        n_jobs = -1
        pipe_custom = mpr.make_mapper_pipeline(
            filter_func=filter_func,
            cover=cover,
            clusterer=clusterer,
            verbose=False,
            n_jobs=n_jobs,
            min_intersection=1
        )
        # Create Mapper graph
        mapper_graph = pipe_custom.fit_transform(X)
        # Create Mapper figure
        plotly_params = {"node_trace": {"marker_colorscale": "jet"}}
        fig = mpr.plot_static_mapper_graph(
            pipe_custom,
            X,
            color_data=y,
            layout_dim=2,
            plotly_params=plotly_params,
            layout="fruchterman_reingold",
            node_scale=60
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
            "./mapper_applications/figures/mapper_diabetes_"
            + f"{clusterer_name}_{n_intervals}_intervals_{overlap_frac}"
            + "_overlap.svg"
        )
        fig.write_image(filename)
