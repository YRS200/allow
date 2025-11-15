"""Hungarian algorithm implementation."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def hungarian(cost: np.ndarray) -> Tuple[List[int], List[int]]:
    """Compute optimal assignment using the Hungarian algorithm."""

    from math import inf

    cost = cost.copy()
    n_rows, n_cols = cost.shape
    dim = max(n_rows, n_cols)
    padded = np.full((dim, dim), cost.max() if cost.size else 0.0, dtype=np.float32)
    padded[:n_rows, :n_cols] = cost

    u = np.zeros(dim + 1)
    v = np.zeros(dim + 1)
    ind = np.zeros(dim + 1, dtype=int)

    for i in range(1, dim + 1):
        links = np.zeros(dim + 1, dtype=int)
        mins = np.full(dim + 1, inf)
        visited = np.zeros(dim + 1, dtype=bool)
        marked_i = i
        marked_j = 0
        j = 0
        while True:
            j = 0
            visited[marked_j] = True
            i0 = ind[marked_j]
            delta = inf
            j1 = 0
            for j in range(1, dim + 1):
                if visited[j]:
                    continue
                cur = padded[marked_i - 1, j - 1] - u[marked_i] - v[j]
                if cur < mins[j]:
                    mins[j] = cur
                    links[j] = marked_j
                if mins[j] < delta:
                    delta = mins[j]
                    j1 = j
            for j in range(dim + 1):
                if visited[j]:
                    u[ind[j]] += delta
                    v[j] -= delta
                else:
                    mins[j] -= delta
            marked_j = j1
            marked_i = ind[marked_j]
            if marked_i == 0:
                break
        while True:
            j1 = links[marked_j]
            ind[marked_j] = ind[j1]
            marked_j = j1
            if marked_j == 0:
                break
        ind[marked_j] = i

    assignment = [(-1, -1)] * dim
    for j in range(1, dim + 1):
        if ind[j] != 0:
            assignment[j - 1] = (ind[j] - 1, j - 1)

    rows = []
    cols = []
    for r, c in assignment:
        if r < n_rows and c < n_cols:
            rows.append(r)
            cols.append(c)
    return rows, cols


__all__ = ["hungarian"]
