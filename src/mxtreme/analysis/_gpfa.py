"""Gaussian Process Factor Analysis -- the pure, arrays-in/arrays-out layer behind
:func:`mxtreme.analysis.network.gpfa`.

The model (Yu et al. 2009, *J Neurophysiol* 102:614-635), for ``q`` channels and ``p`` latents::

    y[:, t] = C x[:, t] + d + eps,   eps ~ N(0, R),  R diagonal
    x[i, :] ~ GP(0, K_i),            K_i(t1, t2) = sf2 exp(-(t1 - t2)^2 / (2 tau_i^2)) + sn2 delta

``C`` (q x p), ``d`` (q) and ``R`` (q) are free; of the kernel, only the timescale ``tau_i`` is
learned. The scale of the latents is fixed by ``K_i(t, t) = 1``, i.e. ``sf2 = 1 - sn2`` with ``sn2``
held at a small constant -- without that the model is free to shrink ``x`` and grow ``C``.

Fitting is EM. The one structural trick that makes it usable on an hour of continuous recording:
write the latents of a segment time-major, ``x_tilde = [x[:, 1]; ...; x[:, T]]``, and the posterior is
``N(Sigma rhs_n, Sigma)`` with ``Sigma = (K^-1 + Lambda)^-1``, where ``Lambda`` is block diagonal with
``T`` copies of ``C' R^-1 C``. **``Sigma`` does not depend on the data**, so all equal-length segments
share it: one Cholesky of the ``pT x pT`` precision per EM iteration serves every segment, and their
posterior means are a single batched triangular solve. The per-iteration cost is therefore
``O((pT)^3)`` *regardless of how many segments there are* -- which is why short segments are better
here, the opposite of the usual intuition.

Everything in this module works in bins, takes ``bin_sec`` where it needs seconds, and knows nothing
about :class:`~mxtreme.recording.Recording`. The three entry points are :func:`fit` (EM on
non-overlapping segments), :func:`latents` (posterior means for every bin, stitched) and
:func:`orthonormalize` (the SVD of ``C`` that makes the latents orderable and plottable).
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar

from mxtreme.params import GPFAParams

# Segments processed at a time when gathering windows and solving for their means. Bounds the peak
# temporary at a few tens of MB without making the batched solve too small to be worth batching.
_SEG_CHUNK = 64

# Tolerance of the timescale search, in log-seconds: 0.02 is ~2% in tau, reached in ~10 evaluations.
_TAU_XATOL = 0.02

# Ratio between the largest and smallest starting timescale is this squared; see _init_taus.
_TAU_INIT_SPREAD = 3.0


# --- kernel and the shared posterior --------------------------------------------------------------


def _sq_dists(n_bins: int, bin_sec: float) -> np.ndarray:
    """``(t1 - t2)**2`` in seconds squared, for every pair of ``n_bins`` bins."""
    t = np.arange(n_bins, dtype=np.float64) * float(bin_sec)
    diff = t[:, None] - t[None, :]
    return diff * diff


def _kernel(tau_sec: float, sq_dists: np.ndarray, sigma_n_sq: float) -> np.ndarray:
    """Squared-exponential kernel with unit diagonal: ``(1 - sn2) exp(-d2 / 2 tau^2) + sn2 I``."""
    k = (1.0 - sigma_n_sq) * np.exp(-sq_dists / (2.0 * float(tau_sec) ** 2))
    k[np.diag_indices_from(k)] += sigma_n_sq
    return k


def _cholesky(matrix: np.ndarray, what: str):
    """``cho_factor`` of ``matrix``, with its log determinant and a message naming the caller."""
    try:
        factor = cho_factor(matrix, lower=True)
    except LinAlgError as exc:
        raise ValueError(f"{what} is not positive definite ({exc}). A timescale far above the bin "
                         "width makes the kernel rank-deficient: lower tau_bounds_sec, raise "
                         "bin_sec, or raise sigma_n_sq.") from None
    return factor, 2.0 * float(np.sum(np.log(np.abs(np.diag(factor[0])))))


def _prior_precision(taus, sq_dists: np.ndarray, sigma_n_sq: float) -> tuple[np.ndarray, float]:
    """``(K_i^-1 for each latent, sum of log|K_i|)`` -- the GP prior, one independent block per dim."""
    inverses, logdet = [], 0.0
    identity = np.eye(sq_dists.shape[0])
    for i, tau in enumerate(taus):
        factor, log_k = _cholesky(_kernel(tau, sq_dists, sigma_n_sq), f"GP kernel of latent {i + 1}")
        inverses.append(cho_solve(factor, identity))
        logdet += log_k
    return np.stack(inverses), logdet


def _precision(kinvs: np.ndarray, ctrinvc: np.ndarray) -> np.ndarray:
    """The ``pT x pT`` posterior precision ``K^-1 + Lambda``, in time-major order (``t * p + i``).

    ``K^-1`` is block diagonal one-latent-at-a-time and ``Lambda`` is block diagonal
    one-time-bin-at-a-time, so each is strided into the other's layout rather than built densely.
    """
    p, n_bins = kinvs.shape[0], kinvs.shape[1]
    prec = np.zeros((n_bins * p, n_bins * p))
    rows = np.arange(n_bins) * p
    for i in range(p):
        index = rows + i
        prec[np.ix_(index, index)] = kinvs[i]
    bins = np.arange(n_bins)
    prec.reshape(n_bins, p, n_bins, p)[bins, :, bins, :] += ctrinvc
    return prec


def _posterior(kinvs: np.ndarray, C: np.ndarray, R: np.ndarray, *, per_dim: bool) -> dict:
    """The posterior covariance of one segment's latents, shared by every segment of that length.

    :param per_dim: Also return each latent's own ``T x T`` covariance block, which the timescale
        update needs and nothing else does.
    :returns: ``{"chol", "logdet_prec", "sum_sigma_tt", "sigma_dim"}`` -- the Cholesky factor of the
        precision (for the means), its log determinant (for the likelihood), the sum over time bins
        of the ``p x p`` diagonal blocks (for the M-step), and the per-latent blocks or ``None``.
    """
    p, n_bins = kinvs.shape[0], kinvs.shape[1]
    ctrinvc = C.T @ (C / R[:, None])
    chol, logdet_prec = _cholesky(_precision(kinvs, ctrinvc), "posterior precision")
    sigma = cho_solve(chol, np.eye(n_bins * p))

    bins = np.arange(n_bins)
    sum_sigma_tt = sigma.reshape(n_bins, p, n_bins, p)[bins, :, bins, :].sum(axis=0)
    sigma_dim = None
    if per_dim:
        rows = bins * p
        sigma_dim = np.stack([sigma[np.ix_(rows + i, rows + i)] for i in range(p)])
    return {"chol": chol, "logdet_prec": logdet_prec, "sum_sigma_tt": sum_sigma_tt,
            "sigma_dim": sigma_dim}


def _rhs(segments: np.ndarray, C: np.ndarray, R: np.ndarray, d: np.ndarray) -> np.ndarray:
    """``C' R^-1 (y - d)`` of every bin, stacked time-major: ``(n_bins * p, n_segments)``.

    :param segments: ``(q, n_segments, n_bins)``.
    """
    q, n_seg, n_bins = segments.shape
    weights = (C / R[:, None]).T                                   # (p, q)
    z = weights @ segments.reshape(q, -1) - (weights @ d)[:, None]  # (p, n_seg * n_bins)
    return np.ascontiguousarray(z.reshape(-1, n_seg, n_bins).transpose(2, 0, 1).reshape(-1, n_seg))


def _means(chol, rhs: np.ndarray, p: int) -> np.ndarray:
    """Posterior mean latents of each segment: ``(p, n_segments, n_bins)`` from a time-major ``rhs``."""
    n_seg = rhs.shape[1]
    solved = cho_solve(chol, rhs)                                   # (n_bins * p, n_seg)
    return np.ascontiguousarray(solved.reshape(-1, p, n_seg).transpose(1, 2, 0))


def _time_major(means: np.ndarray) -> np.ndarray:
    """``means`` back in the ``(n_bins * p, n_segments)`` layout :func:`_rhs` returns, so the two can
    be contracted into the likelihood's ``rhs' mean`` term."""
    n_seg = means.shape[1]
    return means.transpose(2, 0, 1).reshape(-1, n_seg)


def _loglik(n_seg: int, n_bins: int, R: np.ndarray, logdet_k: float, logdet_prec: float,
            quadratic: float, inner: float) -> float:
    """Marginal log likelihood of every segment, from the factorization the E-step already built.

    The marginal covariance of one segment is ``C_big K C_big' + R_big``. Its determinant comes from
    the matrix-determinant lemma, ``|R_big| |K| |K^-1 + C_big' R_big^-1 C_big|``, and its quadratic
    form from Woodbury, which turns into ``(y - d)' R^-1 (y - d) - rhs' mean`` -- both already
    computed. So the likelihood costs nothing beyond what the E-step needed anyway.

    :param quadratic: ``sum (y - d)' R^-1 (y - d)`` over every bin of every segment.
    :param inner: ``sum rhs' mean`` over segments.
    """
    q = R.size
    logdet = n_bins * float(np.sum(np.log(R))) + logdet_k + logdet_prec
    return -0.5 * (n_seg * (q * n_bins * np.log(2.0 * np.pi) + logdet) + quadratic - inner)


# --- M step ---------------------------------------------------------------------------------------


def _mstep_obs(cross: np.ndarray, second: np.ndarray, sum_yy: np.ndarray, n_total: int,
               min_private_var: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form update of ``C``, ``d`` and ``R``, as in factor analysis.

    :param cross: ``sum y [x; 1]'`` over every bin, ``(q, p + 1)``.
    :param second: ``sum E[[x; 1][x; 1]']`` over every bin, ``(p + 1, p + 1)``.
    :param sum_yy: ``sum y * y`` over every bin, ``(q,)`` -- only ``R``'s diagonal is needed.
    :returns: ``(C, d, R)``.
    """
    loadings = np.linalg.solve(second, cross.T).T                   # (q, p + 1) == [C d]
    C, d = loadings[:, :-1], loadings[:, -1]
    variances = (sum_yy - np.einsum("qk,qk->q", loadings, cross)) / n_total
    return C, d, np.maximum(variances, min_private_var)


def _tau_objective(log_tau: float, moment: np.ndarray, n_seg: int, sq_dists: np.ndarray,
                   sigma_n_sq: float) -> float:
    """Negative GP prior term of one latent at ``exp(log_tau)``: ``0.5 (N log|K| + tr(K^-1 S))``."""
    factor, logdet = _cholesky(_kernel(np.exp(log_tau), sq_dists, sigma_n_sq), "GP kernel")
    return 0.5 * (n_seg * logdet + float(np.trace(cho_solve(factor, moment))))


def _mstep_tau(moment: np.ndarray, n_seg: int, sq_dists: np.ndarray, sigma_n_sq: float,
               bounds: tuple[float, float], current: float) -> float:
    """Timescale of one latent that maximises its GP prior term, given the posterior second moment.

    A bounded scalar search over ``log tau``. The result is only accepted if it beats the current
    value, so the EM likelihood cannot go down even when the search stops early.
    """
    args = (moment, n_seg, sq_dists, sigma_n_sq)
    low, high = float(bounds[0]), float(bounds[1])
    found = minimize_scalar(_tau_objective, bounds=(np.log(low), np.log(high)), args=args,
                            method="bounded", options={"xatol": _TAU_XATOL})
    tau = float(np.exp(found.x))
    return tau if found.fun < _tau_objective(np.log(current), *args) else current


# --- initialisation, fit, extraction --------------------------------------------------------------


def _init(segments: np.ndarray, n_latents: int, params: GPFAParams
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Probabilistic-PCA start for ``C``, ``d`` and ``R``: the leading eigenvectors of the sample
    covariance, scaled by the variance they carry above the average of the ones left out.

    The covariance is taken channels-side, as in :func:`mxtreme.analysis.network._pca` -- the channel
    count is at most a few hundred while the bin count runs to tens of thousands.
    """
    q = segments.shape[0]
    flat = segments.reshape(q, -1)
    d = flat.mean(axis=1)
    centered = flat - d[:, None]
    cov = (centered @ centered.T) / max(flat.shape[1] - 1, 1)

    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals, eigvecs = np.clip(eigvals[::-1], 0.0, None), eigvecs[:, ::-1]
    residual = float(eigvals[n_latents:].mean()) if n_latents < q else 0.0
    scale = np.sqrt(np.maximum(eigvals[:n_latents] - residual, 1e-8))
    C = eigvecs[:, :n_latents] * scale
    # A tiny deterministic jitter: with repeated eigenvalues the leading eigenvectors are arbitrary,
    # and an exactly degenerate C makes C' R^-1 C singular on the first iteration.
    C = C + np.random.default_rng(params.seed).normal(scale=1e-6 * (scale.max() or 1.0), size=C.shape)
    R = np.maximum(np.diag(cov) - np.sum(C * C, axis=1), params.min_private_var)
    return C, d, R


def _init_taus(params: GPFAParams, p: int) -> np.ndarray:
    """Starting timescales: log-spaced around ``tau_init_sec`` rather than all equal.

    Equal timescales are a saddle. The model is then invariant to rotating the latents into each
    other, so there is no gradient to separate them, and EM crawls along the plateau -- measured on
    synthetic two-timescale data, 0.3 s and 0.08 s stayed stuck together near 0.09 s for hundreds of
    iterations from an equal start, and separated within ~100 from a spread one.
    """
    if p < 2:
        return np.full(p, float(params.tau_init_sec))
    taus = float(params.tau_init_sec) * _TAU_INIT_SPREAD ** np.linspace(-1.0, 1.0, p)
    return np.clip(taus, *params.tau_bounds_sec)


def pad_bins(params: GPFAParams, taus, n_bins_per_seg: int, bin_sec: float) -> int:
    """How many bins of context to fit either side of a segment when extracting its latents.

    Each segment's GP prior pulls its own edges toward zero, so stitching bare segments leaves a
    visible dip at every seam. Extracting each segment inside a padded window and keeping the middle
    removes it. The default reaches four timescales out, where the prior correlation is negligible.
    """
    overlap = params.overlap_sec
    if overlap is None:
        overlap = 4.0 * float(np.max(taus)) if len(taus) else 0.0
    return int(min(round(overlap / bin_sec), n_bins_per_seg // 4))


def fit(counts: np.ndarray, params: GPFAParams, *, n_latents: int, bin_sec: float,
        n_bins_per_seg: int, verbose: bool = True) -> dict:
    """Fit GPFA by EM to ``counts`` cut into non-overlapping segments.

    :param counts: ``(q, n_bins)``, already transformed; ``n_bins`` must be a whole number of
        segments.
    :param params: Settings; see :class:`~mxtreme.params.GPFAParams`.
    :param n_latents: Number of latent dimensions ``p``, clamped to the channel count.
    :param bin_sec: Width of a bin, in seconds -- the units the timescales come back in.
    :param n_bins_per_seg: Bins per segment ``T``.
    :param verbose: Print the shape of the problem and the likelihood as it goes.
    :returns: ``{"C", "d", "R", "tau_sec", "log_likelihood", "n_iter", "converged", "n_segments"}``.
    """
    q, n_total = counts.shape
    p = max(min(int(n_latents), q), 0)
    n_seg = n_total // n_bins_per_seg
    if q == 0 or p == 0 or n_seg == 0:
        return {"C": np.zeros((q, p)), "d": np.zeros(q), "R": np.ones(q),
                "tau_sec": np.full(p, np.nan), "log_likelihood": np.array([]), "n_iter": 0,
                "converged": False, "n_segments": n_seg}

    used = counts[:, :n_seg * n_bins_per_seg]
    segments = used.reshape(q, n_seg, n_bins_per_seg)
    sq_dists = _sq_dists(n_bins_per_seg, bin_sec)
    n_bins_total = n_seg * n_bins_per_seg
    sum_y = used.sum(axis=1)
    sum_yy = np.einsum("qt,qt->q", used, used)
    if verbose:
        print(f"  GPFA: {q} channels, {n_seg} segments of {n_bins_per_seg} bins, p = {p} "
              f"(pT = {p * n_bins_per_seg})")
        if p * n_bins_per_seg > 4000:
            print(f"  pT = {p * n_bins_per_seg} is large; each EM iteration factorizes a "
                  f"{p * n_bins_per_seg}-square matrix. Lower segment_sec to speed this up.")

    C, d, R = _init(segments, p, params)
    taus = _init_taus(params, p)
    trace: list[float] = []
    converged = False
    # The likelihood scales with the data, so the stopping rule is per channel-bin: "the fit is no
    # longer improving by tol nats per observation". A relative test on the likelihood itself stops
    # far too early -- at a magnitude of 1e5 nats, a 1e-6 relative change is a third of a nat.
    scale = q * n_bins_total

    for iteration in range(int(params.max_iter)):
        learn_tau = bool(params.learn_tau) and (iteration % max(int(params.tau_every), 1) == 0)
        kinvs, logdet_k = _prior_precision(taus, sq_dists, params.sigma_n_sq)
        post = _posterior(kinvs, C, R, per_dim=learn_tau)

        rhs = _rhs(segments, C, R, d)
        means = _means(post["chol"], rhs, p)                        # (p, n_seg, T)
        # Expanded from sum (y - d)' R^-1 (y - d), so no centred copy of the data is made per
        # iteration -- on an hour-long recording that copy is hundreds of MB.
        quadratic = float(np.sum((sum_yy - 2.0 * d * sum_y + n_bins_total * d * d) / R))
        trace.append(_loglik(n_seg, n_bins_per_seg, R, logdet_k, post["logdet_prec"], quadratic,
                             float(np.einsum("in,in->", rhs, _time_major(means)))))

        # Sufficient statistics over every (segment, bin), with the latents augmented by a constant
        # row so C and d come out of a single least-squares solve.
        augmented = np.vstack([means.reshape(p, -1), np.ones((1, n_bins_total))])
        cross = used @ augmented.T
        second = augmented @ augmented.T
        second[:p, :p] += n_seg * post["sum_sigma_tt"]
        C, d, R = _mstep_obs(cross, second, sum_yy, n_bins_total, params.min_private_var)

        if learn_tau:
            for i in range(p):
                moment = n_seg * post["sigma_dim"][i] + means[i].T @ means[i]
                taus[i] = _mstep_tau(moment, n_seg, sq_dists, params.sigma_n_sq,
                                     params.tau_bounds_sec, taus[i])
        if verbose and (iteration % 10 == 0 or iteration == int(params.max_iter) - 1):
            print(f"    iter {iteration + 1}: ll = {trace[-1]:.6g}, tau = "
                  f"{np.array2string(taus, precision=3)}")
        if len(trace) > 1 and (trace[-1] - trace[-2]) <= params.tol * scale:
            converged = True
            break

    low, high = params.tau_bounds_sec
    at_bound = [f"{t:.3g}" for t in taus if min(abs(t - low) / low, abs(t - high) / high) < 0.01]
    if at_bound:
        print(f"  warning: timescale(s) {', '.join(at_bound)} s sat on tau_bounds_sec "
              f"{params.tau_bounds_sec}; the fit wanted to go past the bound, so read them as a "
              "limit rather than a measurement.")
    return {"C": C, "d": d, "R": R, "tau_sec": taus, "log_likelihood": np.asarray(trace),
            "n_iter": len(trace), "converged": converged, "n_segments": n_seg}


def latents(counts: np.ndarray, model: dict, params: GPFAParams, *, bin_sec: float,
            n_bins_per_seg: int, pad: int) -> np.ndarray:
    """Posterior mean latents for every bin of ``counts``, stitched across segments.

    Each segment is inferred inside a window padded by ``pad`` bins on both sides and only its middle
    kept, so a segment's neighbours inform its edges (see :func:`pad_bins`). The very first and last
    windows have no neighbour to borrow from and are padded with zeros instead.

    :returns: ``(p, n_bins)`` float64, aligned bin for bin with ``counts``.
    """
    C, d, R = model["C"], model["d"], model["R"]
    n_total = counts.shape[1]
    p = C.shape[1]
    n_seg = n_total // n_bins_per_seg
    if p == 0 or n_seg == 0:
        return np.zeros((p, n_seg * n_bins_per_seg))

    width = n_bins_per_seg + 2 * pad
    kinvs, _ = _prior_precision(model["tau_sec"], _sq_dists(width, bin_sec), params.sigma_n_sq)
    post = _posterior(kinvs, C, R, per_dim=False)

    padded = np.pad(counts[:, :n_seg * n_bins_per_seg], ((0, 0), (pad, pad)))
    out = np.empty((p, n_seg * n_bins_per_seg))
    offsets = np.arange(width)
    for lo in range(0, n_seg, _SEG_CHUNK):
        hi = min(lo + _SEG_CHUNK, n_seg)
        starts = np.arange(lo, hi) * n_bins_per_seg
        windows = padded[:, starts[:, None] + offsets[None, :]]     # (q, n_chunk, width)
        means = _means(post["chol"], _rhs(windows, C, R, d), p)
        out[:, lo * n_bins_per_seg:hi * n_bins_per_seg] = \
            means[:, :, pad:pad + n_bins_per_seg].reshape(p, -1)
    return out


def orthonormalize(C: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                           np.ndarray]:
    """Rewrite ``C X`` with an orthonormal ``C``, so the latents are ordered and comparable.

    ``C = U S V'`` gives ``C X = U (S V' X)``, which leaves the model's predictions untouched but
    orders the latents by the share of shared variance they carry, as PCA's components are (Yu et al.
    2009, "orthonormalized neural trajectories"). Without it the latents are only defined up to a
    ``p x p`` transform, and two latents with similar timescales can rotate freely into each other.

    Each column's sign is fixed so its loadings sum to a positive number, the same convention as
    :func:`mxtreme.analysis.network._pca`, so one culture doesn't mirror itself between DIVs.

    :returns: ``(C_orth, X_orth, singular_values, shared_var_frac)`` -- the last being each latent's
        share of the shared variance ``trace(C C') = sum s_i**2``.
    """
    C, X = np.asarray(C, dtype=np.float64), np.asarray(X, dtype=np.float64)
    if C.size == 0:
        return C, X, np.zeros(C.shape[1]), np.zeros(C.shape[1])
    u, s, vt = np.linalg.svd(C, full_matrices=False)
    signs = np.where(u.sum(axis=0) < 0, -1.0, 1.0)
    shared = s * s
    total = shared.sum()
    return u * signs, signs[:, None] * (s[:, None] * (vt @ X)), s, (shared / total if total else shared)
