"""Community detection on a functional-connectivity matrix -- the pure, arrays-in/arrays-out layer
behind :func:`mxtreme.analysis.network.communities`.

Every algorithm is a function ``fn(corr, params) -> (labels, modularity)``:

- ``corr`` -- a square, symmetric correlation matrix (its diagonal is ignored);
- ``params`` -- the :class:`~mxtreme.params.CommunityParams` being used;
- ``labels`` -- one integer community id per row of ``corr``, any numbering;
- ``modularity`` -- the partition's quality score, or ``nan`` for a method that has none.

Adding an algorithm means writing one such function and registering it in :data:`METHODS`. The
public wrapper then takes care of everything common to all of them: renumbering the communities
largest-first, the display order and block boundaries, and the channel mapping.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from mxtreme.params import CommunityParams

# Smallest modularity gain worth a move: anything below is float noise, and accepting it can make the
# local-move phase oscillate between two equal partitions.
_EPS = 1e-10


def _weights(corr, negative: str) -> tuple[np.ndarray, np.ndarray | None]:
    """``(positive, negative)`` weight matrices of ``corr``, with no self-loops. ``negative`` is
    ``None`` in ``"clip"`` mode, where anticorrelation carries no weight."""
    w = np.array(corr, dtype=np.float64)
    np.fill_diagonal(w, 0.0)
    w = np.nan_to_num(w, nan=0.0)
    if negative == "clip":
        return np.clip(w, 0.0, None), None
    if negative == "signed":
        return np.clip(w, 0.0, None), np.clip(-w, 0.0, None)
    raise ValueError(f"negative must be 'clip' or 'signed', not {negative!r}")


def modularity_matrix(corr, resolution: float = 1.0, negative: str = "clip") -> np.ndarray:
    """Modularity matrix ``B`` of ``corr``, scaled so ``Q = sum(B[i, j] for i, j in one community)``.

    ``"clip"`` is Newman's weighted modularity on the positive weights. ``"signed"`` is Rubinov &
    Sporns' (2011) asymmetric form, ``Q = Q⁺ - s⁻/(s⁺ + s⁻) · Q⁻``: positive coupling inside a
    community raises Q and negative coupling inside one lowers it, with the negative term weighted
    down because anticorrelations are the weaker evidence.
    """
    w_pos, w_neg = _weights(corr, negative)
    s_pos = w_pos.sum()
    b = np.zeros_like(w_pos)
    if s_pos > 0:
        k = w_pos.sum(axis=1)
        b += (w_pos - resolution * np.outer(k, k) / s_pos) / s_pos
    if w_neg is not None and w_neg.sum() > 0:
        s_neg = w_neg.sum()
        k = w_neg.sum(axis=1)
        b -= (w_neg - resolution * np.outer(k, k) / s_neg) / (s_pos + s_neg)
    return (b + b.T) / 2  # exact symmetry, so a move's gain is the same read from either side


def modularity(b: np.ndarray, labels) -> float:
    """``Q`` of ``labels`` under the modularity matrix ``b`` (from :func:`modularity_matrix`)."""
    labels = np.asarray(labels)
    return float(b[labels[:, None] == labels[None, :]].sum())


def _local_moves(b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Louvain phase 1: start from singletons and move one node at a time to whichever community
    raises Q most, until a full pass over the nodes moves nothing.

    ``h[i, m]`` holds the summed coupling from node ``i`` to community ``m``, so a move's gain is two
    lookups and accepting it is two column updates -- O(n) per node rather than re-scoring Q.
    """
    n = b.shape[0]
    labels = np.arange(n)
    h = b.copy()  # every node starts alone, so its coupling to community j is b[:, j]
    diag = np.diag(b).copy()
    moved = True
    while moved:
        moved = False
        for i in rng.permutation(n):
            current = labels[i]
            row = h[i]
            # Gain of moving i from `current` into each community; i's own self-coupling is in
            # row[current], so it is added back -- staying put scores zero.
            gain = row - row[current] + diag[i]
            gain[current] = 0.0
            target = int(np.argmax(gain))
            if gain[target] > _EPS:
                h[:, target] += b[:, i]
                h[:, current] -= b[:, i]
                labels[i] = target
                moved = True
    return np.unique(labels, return_inverse=True)[1]


def _louvain_once(b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One Louvain run: local moves, then collapse each community to a node and repeat, until the
    collapsed network stops merging."""
    labels = np.arange(b.shape[0])
    level = b
    while True:
        merged = _local_moves(level, rng)
        n_communities = merged.max() + 1
        labels = merged[labels]
        if n_communities == level.shape[0]:
            return labels
        # Community-to-community coupling: summing B over blocks preserves Q exactly, so the next
        # level optimises the same objective on a smaller matrix.
        onehot = np.zeros((level.shape[0], n_communities))
        onehot[np.arange(level.shape[0]), merged] = 1.0
        level = onehot.T @ level @ onehot


def louvain(corr, params: CommunityParams) -> tuple[np.ndarray, float]:
    """Louvain modularity maximisation (Blondel et al. 2008), best of ``params.n_runs`` runs."""
    n = np.asarray(corr).shape[0]
    if n == 0:
        return np.zeros(0, dtype=int), np.nan
    b = modularity_matrix(corr, params.resolution, params.negative)
    if not np.any(b):
        return np.zeros(n, dtype=int), 0.0  # no coupling at all: nothing to divide

    rng = np.random.default_rng(params.seed)
    best_labels, best_q = None, -np.inf
    for _ in range(max(1, params.n_runs)):
        labels = _louvain_once(b, rng)
        q = modularity(b, labels)
        if q > best_q + _EPS:
            best_labels, best_q = labels, q
    return best_labels, best_q


# Registered algorithms, by the name `CommunityParams.method` selects them with. Add new ones here.
METHODS: dict[str, Callable] = {
    "louvain": louvain,
}


def _largest_first(labels) -> np.ndarray:
    """Renumber ``labels`` so community 0 is the largest (ties broken by first appearance)."""
    labels = np.asarray(labels)
    if labels.size == 0:
        return labels.astype(int)
    ids, first, inverse, counts = np.unique(labels, return_index=True, return_inverse=True,
                                            return_counts=True)
    rank = np.empty(ids.size, dtype=int)
    rank[np.lexsort((first, -counts))] = np.arange(ids.size)  # rank[k]: new id of community ids[k]
    return rank[inverse]


def _leaf_order(corr, members) -> np.ndarray:
    """``members`` reordered by average-linkage clustering on ``1 - r``, so strongly correlated
    electrodes sit next to each other inside their community's block."""
    if members.size < 3:
        return members
    dist = 1.0 - np.asarray(corr, dtype=np.float64)[np.ix_(members, members)]
    dist = np.clip(np.nan_to_num((dist + dist.T) / 2, nan=1.0), 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    return members[leaves_list(linkage(squareform(dist, checks=False), method="average"))]


def detect(corr, params: CommunityParams, with_order: bool = True) -> dict:
    """Run ``params.method`` on ``corr`` and add what every method's result shares.

    :param with_order: Also work out the display order (a clustering per community); off for callers
        that only want the scalars.
    :returns: ``{"labels": community per row (0 = largest), "order": a permutation of the rows that
        groups them by community, largest first, "boundaries": where each community's block ends in
        that order, "modularity": float, "n_communities": int}`` (no ``order`` without
        ``with_order``).
    """
    try:
        method = METHODS[params.method]
    except KeyError:
        raise ValueError(f"Unknown community-detection method {params.method!r}. "
                         f"Available: {sorted(METHODS)}") from None

    corr = np.asarray(corr)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError(f"Expected a square correlation matrix, got shape {corr.shape}.")
    raw, q = method(corr, params)
    labels = _largest_first(raw)
    counts = np.bincount(labels) if labels.size else np.zeros(0, int)
    result = {"labels": labels, "boundaries": np.cumsum(counts), "modularity": float(q),
              "n_communities": int(counts.size)}
    if with_order:
        blocks = [_leaf_order(corr, np.flatnonzero(labels == c)) for c in range(counts.size)]
        result["order"] = np.concatenate(blocks).astype(int) if blocks else np.zeros(0, int)
    return result
