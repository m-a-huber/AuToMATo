Code for the paper called ``AuToMATo: A Parameter-Free Persistence-Based Clustering Algorithm'' by Huber, Kali&#353;nik and Schnider.

To run the scripts reproducing the results and figures in the paper, run ``python3 -m eval.eval``, ``python3 -m mapper_applications.make_diabetes_pictures`` or ``python3 -m mapper_applications.make_synth_pictures``.

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