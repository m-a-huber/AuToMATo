Code for the paper called ``AuToMATo: An Out-Of-The-Box Persistence-Based Clustering Algorithm''.

To run the scripts reproducing the results and figures in the paper, run `python3 -m eval.eval`, `python3 -m mapper_applications.make_diabetes_pictures` or `python3 -m mapper_applications.make_synth_pictures`.

If the TTK-clustering algorithm is to be included in evaluation (uncomment relevant lines in eval.py to do so), ParaView and the Topology ToolKit must be installed.

---

__Example of running AuToMATo__

```
>>> from automato import Automato
>>> from sklearn.datasets import make_blobs
>>> X, y = make_blobs(centers=2, random_state=42)
>>> aut = Automato(random_state=42).fit(X)
>>> aut.n_clusters_
2
>>> (aut.labels_ == y).all()
True
```

---

__Requirements__

Required Python dependencies are specified in `pyproject.toml`. Developer dependencies indicate those that are needed only in order to run the scripts that reproduce results and figures from the paper, as opposed to using AuToMATo on its own. Provided that `uv` is installed, these dependencies can be installed by running `uv pip instal -r pyproject.toml` (to exclude `dev`-dependencies) and `uv pip install --group dev -r pyproject.toml` (to include `dev`-dependencies). The environment specified in `uv.lock` can be recreated by running `uv sync --no-dev` and `uv sync`, respectively.

---

__Example of installing AuToMATo for `uv` users__

```
$ git clone url/to/repo
$ cd AuToMATo
$ uv sync --no-dev
$ source .venv/bin/activate
$ python
>>> from automato import Automato
>>> ...
```
