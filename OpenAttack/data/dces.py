"""
:type: tuple
:Size: 1.295MB

:Package Requirements: * **sklearn**

Vec-colnames and neighber matrix used in Substitute DECS. See :py:class:`.DCESSubstitute` for detail.

"""

import os
import pickle
from OpenAttack.utils import make_zip_downloader

NAME = "AttackAssist.DCES"

URL = "/TAADToolbox/DCES.zip"
DOWNLOAD = make_zip_downloader(URL)


def LOAD(path):
    with open(os.path.join(path, 'descs.pkl'), 'rb') as f:
        descs = pickle.load(f)
    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(** {
        'algorithm': 'auto',
        'leaf_size': 30,
        'metric': 'euclidean',
        'metric_params': None,
        'n_jobs': 1,
        'n_neighbors': 5,
        'p': 2,
        'radius': 1.0
    })
    return descs, neigh
