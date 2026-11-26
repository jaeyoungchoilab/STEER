"""Embedding similarity (ES): the cosine between two proteins' centred mean embeddings.

The embeddings are centred by subtracting a mean vector before the cosine is taken. Without centring, every pair of
ESM-2 mean embeddings has a cosine close to 1 (they share a large common component), so centring is what makes ES
informative. By default the centre is the mean over the proteins being compared, as in the paper, where each
proteome (or orthogroup set) was centred on itself. ES therefore depends on the protein set: values from a small or
unusual set are not comparable with values from a whole proteome. To score a few proteins on the scale of a
proteome, centre them with that proteome's mean (`center=` an array, or --center FILE in the command-line tool).
"""
import numpy as np


def center_vector(V):
    """The mean of the rows of V (the default centre)."""
    return np.asarray(V, np.float64).mean(0)


def unit_rows(V, center=None):
    """Rows of V minus the centre, scaled to length 1 (float32). center: None = mean of V's rows, False = none,
    or a vector."""
    V = np.asarray(V, np.float64)
    if center is None:
        V = V - center_vector(V)
    elif center is not False:
        V = V - np.asarray(center, np.float64)
    return (V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def es_matrix(V, center=None, out=None, dtype=np.float32, block=2000):
    """n x n ES matrix (diagonal set to NaN). With out (a .npy path) the matrix is written block by block to a
    memory-mapped file and returned as that memmap, so matrices larger than memory can be written."""
    U = unit_rows(V, center)
    n = len(U)
    M = (np.lib.format.open_memmap(out, mode="w+", dtype=dtype, shape=(n, n)) if out
         else np.empty((n, n), dtype))
    for r0 in range(0, n, block):
        r1 = min(r0 + block, n)
        B = U[r0:r1] @ U.T
        B[np.arange(r1 - r0), np.arange(r0, r1)] = np.nan
        M[r0:r1] = B.astype(dtype)
    if out:
        M.flush()
    return M


def es_pair(V, i, j, center=None):
    """ES of one pair (rows i and j of V)."""
    U = unit_rows(V, center)
    return float(U[i] @ U[j])


def random_pair_values(M, n_pairs=2_000_000, seed=1):
    """Sorted ES values of random distinct pairs: the background for 'top %' (share of pairs with ES at least
    this high)."""
    n = M.shape[0]
    rng = np.random.default_rng(seed)
    i, j = rng.integers(0, n, n_pairs), rng.integers(0, n, n_pairs)
    keep = i != j
    v = np.asarray(M[i[keep], j[keep]], np.float64)
    return np.sort(v[np.isfinite(v)])


def top_percent(values, background):
    """Percentage of background pairs with ES >= each value (small = among the most similar pairs)."""
    values = np.asarray(values, np.float64)
    return 100 * (len(background) - np.searchsorted(background, values, side="left")) / len(background)


def top_partners(M, names, k=20, background=None):
    """Rows (protein, rank, partner, ES[, top_percent]) for each protein's k highest-ES partners."""
    rows = []
    for a in range(M.shape[0]):
        r = np.asarray(M[a], np.float64)
        r = np.where(np.isfinite(r), r, -np.inf)
        best = np.argsort(-r, kind="stable")[:k]
        tp = top_percent(r[best], background) if background is not None else [None] * len(best)
        for rank, (b, t) in enumerate(zip(best, tp), 1):
            if np.isfinite(r[b]):
                rows.append((names[a], rank, names[b], float(r[b])) + (() if t is None else (float(t),)))
    return rows
