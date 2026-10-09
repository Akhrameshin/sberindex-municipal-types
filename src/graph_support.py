"""Weighted graphs preserve support even when two profiles have zero distance."""
import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


def weighted_knn(x, metric="cosine", k=15, minimum=3, mutual=True, eps=False):
    n = len(x)
    if n < 2:
        return sparse.csr_matrix((n, n), dtype=float)
    k = min(int(k), n - 1)
    minimum = min(int(minimum), n - 1)
    query = min(max(k, minimum) + 1, n)
    if metric == 'precomputed':
        matrix=np.asarray(x,dtype=float).copy()
        if matrix.shape != (n,n):raise ValueError('Distance matrix must be square')
        matrix[~np.isfinite(matrix)]=np.inf
        np.fill_diagonal(matrix,-np.inf)
        indices=np.argsort(matrix,axis=1,kind='stable')[:,:query]
        distances=np.take_along_axis(matrix,indices,axis=1)
    else:
        model = NearestNeighbors(n_neighbors=query, metric=metric).fit(x)
        distances, indices = model.kneighbors(x)
    rows, cols, vals, nr, nc, nv = [], [], [], [], [], []
    for i in range(n):
        take = indices[i] != i
        js, ds = indices[i][take], distances[i][take]
        good = np.isfinite(ds)
        js, ds = js[good], np.maximum(ds[good], 0)
        for j, d in zip(js[:k], ds[:k]):
            rows.append(i); cols.append(j); vals.append(1 / (d + 1e-3))
        for j, d in zip(js[:minimum], ds[:minimum]):
            nr.append(i); nc.append(j); nv.append(1 / (d + 1e-3))
    a = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    if eps:
        from sklearn.metrics import pairwise_distances
        distance = np.asarray(x) if metric == "precomputed" else pairwise_distances(x, metric=metric)
        upper = distance[np.triu_indices(n, 1)]
        finite=upper[np.isfinite(upper)]
        threshold = np.quantile(finite, min(1, k / (n - 1))) if finite.size else -np.inf
        support = np.isfinite(distance) & (distance <= threshold)
        np.fill_diagonal(support, False)
        ii, jj = np.nonzero(support)
        a = sparse.csr_matrix((1 / (np.maximum(distance[ii, jj], 0) + 1e-3), (ii, jj)), shape=(n, n))
    else:
        a = a.minimum(a.T) if mutual else a.maximum(a.T)
    near = sparse.csr_matrix((nv, (nr, nc)), shape=(n, n))
    a = a.maximum(near.maximum(near.T)).tocsr()
    a.setdiag(0); a.eliminate_zeros()
    if a.data.size:
        a.data /= a.data.mean()
    return a
