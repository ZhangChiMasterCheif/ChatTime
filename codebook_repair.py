"""
Repaired codebook-distance CQR: a real-valued coverage guarantee.

The problem the reviewers identified
------------------------------------
Split conformal on token indices guarantees, for the inflated endpoint tokens
b_lo and b_hi,

    P( tok(y) in [b_lo, b_hi]  for all h )  >=  1 - alpha .            (token)

Eq. 18 of the submission then returns the interval

    [ center(b_lo) * sigma ,  center(b_hi) * sigma ] .

Tokenization is many-to-one: token b is the preimage of the half-open cell
(edge[b], edge[b+1]]. So `tok(y) == b_hi` only puts y somewhere in that cell,
and y can sit anywhere up to edge[b_hi + 1] -- which is strictly above
center(b_hi). The token-space event therefore does NOT imply the value-space
event, and (token) does not transfer to prediction space. The gap is up to half
a bin width at each end, and it is unbounded at the extreme tokens, whose
preimages are (-inf, edge_1] and (edge_{L-1}, +inf).

Using bin EDGES instead fixes the interior but makes the interval infinite the
moment an endpoint lands on an extreme token, so coverage and boundedness are
not jointly established. That is exactly the reviewers' objection, and it is
correct.

The repair
----------
Replace the integer bin index by a strictly increasing, piecewise-linear
embedding of the whole real line onto bin coordinates:

    phi(z) = b + (z - edge[b]) / (edge[b+1] - edge[b])    for z in cell b,

with the two unbounded end cells extrapolated linearly using the width of
their nearest finite neighbour. phi is a continuous, strictly increasing
bijection R -> R, with phi(edge[b]) = b, so one unit of phi is exactly one bin
-- the metric the submission wanted -- but now resolved fractionally rather
than rounded.

Run ordinary CQR in phi coordinates and map the endpoints back through
phi^{-1}. Because phi is a strictly increasing bijection,

    phi(y) in [phi(q_lo) - Q, phi(q_hi) + Q]
        <=>  y in [phi^{-1}(phi(q_lo) - Q), phi^{-1}(phi(q_hi) + Q)]

is an EQUIVALENCE, not an implication. Value-space coverage therefore equals
phi-space coverage exactly, and the split-conformal guarantee transfers with no
support assumption and no loss:

    P( y_h in [l_h, u_h] for all h )  >=  1 - alpha  .

phi^{-1} is finite everywhere, so the returned interval is always finite. The
three properties claimed in Section 4.5 survive:

  * bounded          -- the interval is finite for every input, and any target
                        inside the codebook range has |score| <= L;
  * scale invariant  -- sigma is divided out before embedding;
  * model-aligned    -- a unit of phi is one codebook bin.

Four endpoint conventions are implemented so the rebuttal can show the cost of
each side by side:

    "center"        as submitted. Token-space guarantee only.
    "edge"          bin edges. Valid in value space, infinite at extreme tokens.
    "edge_clipped"  bin edges with the outer edges replaced by the finite
                    codebook limits. Finite, and valid CONDITIONAL on the target
                    lying inside the codebook range -- a checkable assumption
                    whose empirical rate we report.
    "phi"           the repair. Finite and unconditionally valid.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Tokenizer geometry
# ---------------------------------------------------------------------------

class BinGrid:
    """
    Geometry of a uniform-bin tokenizer, independent of which model produced it.

    Parameters
    ----------
    edges : (L+1,) increasing array of cell boundaries in SCALED space. The
            first and last may be +/-inf (Chronos uses +/-1e20, which we treat
            as infinite). Cell b is (edges[b], edges[b+1]].
    centers : (L,) representative value of each cell, i.e. what the inverse
            tokenizer returns.
    offset : integer added to the cell index to obtain the model's token id
            (Chronos adds n_special_tokens + 1; ChatTime adds 0).

            Recorded for reference only -- every method here works in CELL
            space, re-deriving cells from values rather than consuming the
            model's emitted token ids, so the special-token offset never enters
            the arithmetic. Keep it accurate anyway: it is what you need if you
            ever want to line these cells up against cached `sample_bins`.
    """

    BIG = 1e19  # anything beyond this in |value| is treated as an infinite edge

    def __init__(self, edges, centers, offset=0):
        self.edges = np.asarray(edges, dtype=np.float64)
        self.centers = np.asarray(centers, dtype=np.float64)
        self.offset = int(offset)
        self.L = len(self.centers)
        assert len(self.edges) == self.L + 1, (
            f"expected {self.L + 1} edges for {self.L} centers, got {len(self.edges)}"
        )

        # Finite interior widths, used to extrapolate through the open end cells.
        e = self.edges.copy()
        self.finite = np.abs(e) < self.BIG
        w = np.diff(e)
        w_finite = w[np.isfinite(w) & (np.abs(w) < self.BIG)]
        self.typ_w = float(np.median(w_finite)) if len(w_finite) else 1.0

        # Working edge array with the open ends replaced by a linear
        # extrapolation one typical cell beyond the last finite edge. phi stays
        # strictly increasing and finite across the whole line.
        self.e_work = e.copy()
        if not self.finite[0]:
            self.e_work[0] = e[1] - self.typ_w
        if not self.finite[-1]:
            self.e_work[-1] = e[-2] + self.typ_w

        self.lo_limit = float(self.e_work[0])
        self.hi_limit = float(self.e_work[-1])

    # -- classic tokenization -------------------------------------------------

    def to_cell(self, z):
        """Scaled value -> cell index in [0, L-1]. Matches bucketize(right=True)."""
        idx = np.searchsorted(self.edges, np.asarray(z, dtype=np.float64), side="left") - 1
        return np.clip(idx, 0, self.L - 1)

    def cell_to_center(self, b):
        return self.centers[np.clip(np.asarray(b).astype(int), 0, self.L - 1)]

    def cell_to_lower_edge(self, b):
        return self.edges[np.clip(np.asarray(b).astype(int), 0, self.L - 1)]

    def cell_to_upper_edge(self, b):
        return self.edges[np.clip(np.asarray(b).astype(int) + 1, 0, self.L)]

    # -- monotone embedding ---------------------------------------------------

    def phi(self, z):
        """
        Strictly increasing piecewise-linear map R -> R with phi(e_b) = b.

        Implemented as an interpolation on the working edge array, extended
        linearly with slope 1/typ_w outside it so that phi is defined and
        strictly increasing on the entire real line.
        """
        z = np.asarray(z, dtype=np.float64)
        e = self.e_work
        out = np.interp(z, e, np.arange(self.L + 1, dtype=np.float64))

        below = z < e[0]
        above = z > e[-1]
        if np.any(below):
            out = np.where(below, (z - e[0]) / self.typ_w, out)
        if np.any(above):
            out = np.where(above, self.L + (z - e[-1]) / self.typ_w, out)
        return out

    def phi_inv(self, t):
        """Inverse of `phi`; finite for every finite input."""
        t = np.asarray(t, dtype=np.float64)
        e = self.e_work
        grid = np.arange(self.L + 1, dtype=np.float64)
        out = np.interp(t, grid, e)

        below = t < 0.0
        above = t > self.L
        if np.any(below):
            out = np.where(below, e[0] + t * self.typ_w, out)
        if np.any(above):
            out = np.where(above, e[-1] + (t - self.L) * self.typ_w, out)
        return out

    # -- constructors ---------------------------------------------------------

    @classmethod
    def from_chronos(cls, tokenizer):
        """Build from a live chronos MeanScaleUniformBins tokenizer."""
        centers = tokenizer.centers.detach().cpu().numpy().astype(np.float64)
        edges = tokenizer.boundaries.detach().cpu().numpy().astype(np.float64)
        offset = int(tokenizer.config.n_special_tokens) + 1
        return cls(edges, centers, offset=offset)

    @classmethod
    def from_chronos_arrays(cls, centers, boundaries, n_special):
        return cls(boundaries, centers, offset=int(n_special) + 1)

    @classmethod
    def from_chattime(cls, low=-1.0, high=1.0, n_tokens=10002):
        """
        Rebuild ChatTime's Discretizer geometry (utils/tools.py).

        Discretizer uses `boundaries = linspace(low, high, n_tokens - 1)` and
        `np.digitize(..., right=True)`, then duplicates the first and last
        center so that `centers` lines up with digitize's output range. We
        reproduce that padding here so cell b and token b agree.
        """
        boundaries = np.linspace(low, high, n_tokens - 1)
        centers = (boundaries[1:] + boundaries[:-1]) / 2.0
        centers = np.concatenate((centers[:1], centers, centers[-1:]))
        # digitize(right=True) returns 0 for z <= boundaries[0] and len(b) for
        # z > boundaries[-1], so the cell array has len(boundaries)+1 members
        # and its edges are the boundaries padded with +/-inf.
        edges = np.concatenate(([-np.inf], boundaries, [np.inf]))
        assert len(centers) == len(edges) - 1
        return cls(edges, centers, offset=0)


# ---------------------------------------------------------------------------
# Codebook CQR with selectable endpoint convention
# ---------------------------------------------------------------------------

def _base_quantiles_scaled(samples, alpha, scale):
    """
    Per-step sample quantiles, expressed in SCALED (pre-tokenizer) space.

    Uses np.quantile when the array is NaN-free; np.nanquantile is ~40x slower
    and there is nothing for it to skip on a Chronos cache.
    """
    q = np.nanquantile if bool(np.isnan(samples).any()) else np.quantile
    q_lo = q(samples, alpha / 2.0, axis=1)
    q_hi = q(samples, 1 - alpha / 2.0, axis=1)
    s = np.asarray(scale, dtype=np.float64)[:, None]
    return q_lo / s, q_hi / s


def codebook_cqr(cal_samples, cal_futures, cal_scale,
                 test_samples, test_scale,
                 grid, alpha, endpoint="phi"):
    """
    Codebook-distance CQR under one of four endpoint conventions.

    Parameters
    ----------
    cal_samples, test_samples : (N, K, H) sample paths in VALUE space
    cal_futures               : (N, H)    targets in value space
    cal_scale, test_scale     : (N,)      per-window tokenizer scale
    grid                      : BinGrid
    endpoint                  : "phi" | "edge" | "edge_clipped" | "center"

    Returns
    -------
    lo, hi : (N_test, H) interval bounds in value space
    info   : dict with Q-hat, the fraction of endpoints landing on an extreme
             cell, and the support-assumption violation rate
    """
    if endpoint == "phi":
        return _codebook_cqr_phi(
            cal_samples, cal_futures, cal_scale,
            test_samples, test_scale, grid, alpha,
        )
    return _codebook_cqr_discrete(
        cal_samples, cal_futures, cal_scale,
        test_samples, test_scale, grid, alpha, endpoint,
    )


def _codebook_cqr_phi(cal_s, cal_f, cal_sc, test_s, test_sc, grid, alpha):
    """
    The repair: CQR carried out in phi coordinates.

    Because phi is a strictly increasing bijection of the real line, coverage in
    phi space and coverage in value space are the SAME event, so the standard
    split-conformal argument gives an unconditional real-valued guarantee.
    """
    cal_lo_z, cal_hi_z = _base_quantiles_scaled(cal_s, alpha, cal_sc)
    cal_y_z = np.asarray(cal_f, dtype=np.float64) / np.asarray(cal_sc, np.float64)[:, None]

    p_lo = grid.phi(cal_lo_z)
    p_hi = grid.phi(cal_hi_z)
    p_y = grid.phi(cal_y_z)

    scores = np.maximum(p_lo - p_y, p_y - p_hi).max(axis=1)
    scores = scores[np.isfinite(scores)]
    n = len(scores)
    level = np.ceil((1 - alpha) * (n + 1)) / n
    q = np.inf if level > 1.0 else float(np.quantile(scores, level))

    t_lo_z, t_hi_z = _base_quantiles_scaled(test_s, alpha, test_sc)
    sc = np.asarray(test_sc, dtype=np.float64)[:, None]

    if not np.isfinite(q):
        big = np.full_like(t_lo_z, np.inf)
        return -big, big, {"Q_hat_bins": np.inf, "endpoint": "phi"}

    lo = grid.phi_inv(grid.phi(t_lo_z) - q) * sc
    hi = grid.phi_inv(grid.phi(t_hi_z) + q) * sc
    return lo, hi, {"Q_hat_bins": q, "endpoint": "phi",
                    "frac_endpoint_extreme": 0.0, "frac_support_violation": 0.0}


def _codebook_cqr_discrete(cal_s, cal_f, cal_sc, test_s, test_sc,
                           grid, alpha, endpoint):
    """Integer-bin CQR, decoded with centers, true edges, or clipped edges."""
    cal_lo_z, cal_hi_z = _base_quantiles_scaled(cal_s, alpha, cal_sc)
    cal_y_z = np.asarray(cal_f, dtype=np.float64) / np.asarray(cal_sc, np.float64)[:, None]

    b_lo = grid.to_cell(cal_lo_z)
    b_hi = grid.to_cell(cal_hi_z)
    b_y = grid.to_cell(cal_y_z)

    scores = np.maximum(b_lo - b_y, b_y - b_hi).max(axis=1).astype(np.float64)
    n = len(scores)
    level = np.ceil((1 - alpha) * (n + 1)) / n
    q = np.inf if level > 1.0 else float(np.quantile(scores, level))

    t_lo_z, t_hi_z = _base_quantiles_scaled(test_s, alpha, test_sc)
    t_lo = grid.to_cell(t_lo_z)
    t_hi = grid.to_cell(t_hi_z)
    sc = np.asarray(test_sc, dtype=np.float64)[:, None]

    lo_cell = np.clip(t_lo - q, 0, grid.L - 1).astype(int)
    hi_cell = np.clip(t_hi + q, 0, grid.L - 1).astype(int)

    if endpoint == "center":
        lo = grid.cell_to_center(lo_cell) * sc
        hi = grid.cell_to_center(hi_cell) * sc
    elif endpoint in ("edge", "edge_clipped"):
        lo = grid.cell_to_lower_edge(lo_cell)
        hi = grid.cell_to_upper_edge(hi_cell)
        if endpoint == "edge_clipped":
            # Replace the infinite outer edges by the finite codebook limits.
            # Valid only under the support assumption checked below.
            lo = np.where(np.isfinite(lo) & (np.abs(lo) < grid.BIG), lo, grid.lo_limit)
            hi = np.where(np.isfinite(hi) & (np.abs(hi) < grid.BIG), hi, grid.hi_limit)
        else:
            lo = np.where(np.abs(lo) >= grid.BIG, -np.inf, lo)
            hi = np.where(np.abs(hi) >= grid.BIG, np.inf, hi)
        lo = lo * sc
        hi = hi * sc
    else:
        raise ValueError(f"unknown endpoint convention: {endpoint}")

    info = {
        "Q_hat_bins": q,
        "endpoint": endpoint,
        "frac_endpoint_extreme": float(
            ((lo_cell <= 0) | (hi_cell >= grid.L - 1)).mean()
        ),
    }
    return lo, hi, info


def support_violation_rate(futures, scale, grid):
    """
    Fraction of targets outside the codebook's representable range.

    This is the assumption `edge_clipped` needs, made explicit and measurable.
    It is exactly the quantity the reviewers asked us to state.
    """
    z = np.asarray(futures, dtype=np.float64) / np.asarray(scale, np.float64)[:, None]
    outside = (z <= grid.lo_limit) | (z > grid.hi_limit)
    return {
        "frac_steps_outside_codebook": float(outside.mean()),
        "frac_windows_outside_codebook": float(outside.any(axis=1).mean()),
        "max_joint_coverage_achievable": float(1.0 - outside.any(axis=1).mean()),
    }


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

def _self_test(seed=0, n=4000, K=100, H=4, alpha=0.1):
    """
    Verify on synthetic data that:
      * phi is a strictly increasing bijection (phi_inv(phi(z)) == z);
      * the "phi" variant attains its nominal JOINT coverage;
      * the "center" variant (as submitted) undercovers in value space even
        though its token-space coverage is valid -- i.e. reproduces the exact
        failure the reviewers describe;
      * "edge" is valid but can return infinite endpoints.
    """
    rng = np.random.default_rng(seed)
    grid = BinGrid.from_chronos_arrays(
        centers=np.linspace(-15, 15, 4094),
        boundaries=np.concatenate(
            ([-1e20],
             (np.linspace(-15, 15, 4094)[1:] + np.linspace(-15, 15, 4094)[:-1]) / 2,
             [1e20])
        ),
        n_special=2,
    )

    z = np.linspace(-40, 40, 20001)
    err = np.max(np.abs(grid.phi_inv(grid.phi(z)) - z))
    mono = bool(np.all(np.diff(grid.phi(z)) > 0))
    print(f"phi bijection: max |phi_inv(phi(z)) - z| = {err:.3e}, strictly increasing = {mono}")

    def synth(m):
        scale = rng.uniform(0.5, 50.0, m)
        base = rng.normal(0, 1, (m, 1, H)) * 2.0
        s = (base + rng.normal(0, 1.0, (m, K, H))) * scale[:, None, None]
        f = (base[:, 0, :] + rng.normal(0, 1.0, (m, H))) * scale[:, None]
        return s, f, scale

    cal_s, cal_f, cal_sc = synth(n)
    test_s, test_f, test_sc = synth(n)

    print(f"\n{'endpoint':<14} {'joint cov':>10} {'per-step':>9} {'width':>12} {'inf?':>6}")
    for ep in ("center", "edge", "edge_clipped", "phi"):
        lo, hi, info = codebook_cqr(
            cal_s, cal_f, cal_sc, test_s, test_sc, grid, alpha, endpoint=ep
        )
        cov = (test_f >= lo) & (test_f <= hi)
        w = hi - lo
        finite = np.isfinite(w)
        print(f"{ep:<14} {cov.all(axis=1).mean():>10.4f} {cov.mean():>9.4f} "
              f"{(np.mean(w[finite]) if finite.any() else np.inf):>12.2f} "
              f"{'yes' if not finite.all() else 'no':>6}")

    print(f"\nnominal joint coverage = {1 - alpha:.2f}")
    print("support:", support_violation_rate(test_f, test_sc, grid))


def _stress_test(seed=0, n=4000, K=100, H=4, alpha=0.1,
                 n_bins=64, limit=4.0, tail_df=2.0):
    """
    Reproduce the reviewers' failure mode on purpose.

    Two knobs drive it:
      * a COARSE grid (n_bins), so the half-bin gap between a cell's center and
        its upper edge is no longer negligible;
      * HEAVY-TAILED targets that leave the codebook range, so endpoint tokens
        land on the extreme cells whose preimages are infinite.

    Expected outcome:
      center       undercovers in value space -- the flaw as stated;
      edge         valid, but returns infinite endpoints;
      edge_clipped finite, and undercovers by roughly the support-violation rate;
      phi          valid AND finite.
    """
    rng = np.random.default_rng(seed)
    centers = np.linspace(-limit, limit, n_bins)
    edges = np.concatenate(([-1e20], (centers[1:] + centers[:-1]) / 2, [1e20]))
    grid = BinGrid(edges, centers, offset=0)

    def synth(m):
        scale = np.ones(m)
        base = rng.standard_t(tail_df, (m, 1, H)) * 0.5
        s = base + rng.standard_t(tail_df, (m, K, H)) * 0.5
        f = base[:, 0, :] + rng.standard_t(tail_df, (m, H)) * 0.5
        return s, f, scale

    cal_s, cal_f, cal_sc = synth(n)
    test_s, test_f, test_sc = synth(n)

    sup = support_violation_rate(test_f, test_sc, grid)
    print(f"\n--- stress: {n_bins} bins over [-{limit}, {limit}], t({tail_df}) targets ---")
    print(f"targets outside codebook: {sup['frac_steps_outside_codebook']:.4f} of steps, "
          f"{sup['frac_windows_outside_codebook']:.4f} of windows")
    print(f"=> a truncating method cannot exceed "
          f"{sup['max_joint_coverage_achievable']:.4f} joint coverage\n")

    print(f"{'endpoint':<14} {'joint cov':>10} {'per-step':>9} {'mean width':>12} {'inf?':>6}")
    for ep in ("center", "edge", "edge_clipped", "phi"):
        lo, hi, info = codebook_cqr(
            cal_s, cal_f, cal_sc, test_s, test_sc, grid, alpha, endpoint=ep
        )
        cov = (test_f >= lo) & (test_f <= hi)
        w = hi - lo
        finite = np.isfinite(w)
        wtxt = f"{np.mean(w[finite]):.2f}" if finite.any() else "inf"
        flag = "yes" if not finite.all() else "no"
        print(f"{ep:<14} {cov.all(axis=1).mean():>10.4f} {cov.mean():>9.4f} "
              f"{wtxt:>12} {flag:>6}")
    print(f"\nnominal joint coverage = {1 - alpha:.2f}")


if __name__ == "__main__":
    _self_test()
    _stress_test()
