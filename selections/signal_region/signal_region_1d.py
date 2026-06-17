"""1-D optimal-binning prototype for signal-region definition.

Instead of searching for N non-overlapping rectangles in the full multiclass
score space, this defines a single discriminant

    p(signal) = sum of the signal-class probabilities  (here p(VVV) + p(VH))

and partitions it into N contiguous bins (= signal regions) that maximise the
combined significance Z_comb = sqrt(sum_i Z_i^2), with each bin required to hold
at least ``min_bkg_weight`` background.

The optimal N-bin partition of a 1-D variable under an additive bin objective is
found EXACTLY by dynamic programming over a fixed edge grid -- no beam search,
no branch-and-bound, no diversity tricks. It runs in well under a second once
the data is loaded.

Data loading / model inference / weights are reused verbatim from
signal_region_hist.prepare_inputs (which reuses signal_region.py). Point
``SR_HIST_CONFIG_PATH`` at the same config file used by the box-search tool.

    SR_HIST_CONFIG_PATH=$PWD/config_hist.json python3 ./signal_region_1d.py
"""

import os
import numpy as np

import signal_region_hist as srh  # sets SCAN_CONFIG_PATH and imports signal_region

sr = srh.sr
log_message = sr.log_message

# -------------------- Config --------------------
_cfg = sr._load_json(srh.HIST_CFG_PATH)
N_BINS   = max(1, int(_cfg.get("n_signal_regions", 4)))
MIN_BKG  = float(sr.MIN_BKG_WEIGHT)
EDGE_W   = float(_cfg.get("onedim_bin_width", 0.005))
# Which classes count as "signal" for the discriminant (default: the signal
# classes, i.e. VVV + VH). Override with config key "signal_score_classes".
_sig_cls = _cfg.get("signal_score_classes")
if _sig_cls is None:
    SIGNAL_AXES = list(sr.SIGNAL_CLASS_INDICES)
else:
    SIGNAL_AXES = [sr.CLASS_NAMES.index(c) if isinstance(c, str) else int(c)
                   for c in _sig_cls]
SIGNAL_AXIS_NAMES = [sr.CLASS_NAMES[i] for i in SIGNAL_AXES]
NEG = -1.0e18


def _z2(S, B):
    """Asymptotic Z^2 = 2[(S+B)ln(1+S/B) - S], vectorised, invalid -> NEG."""
    S = np.asarray(S, dtype=float)
    B = np.asarray(B, dtype=float)
    ok = (B >= MIN_BKG) & (S > 0.0)
    Bs = np.where(ok, B, 1.0)
    Ss = np.where(ok, S, 0.0)
    f = (Ss + Bs) * np.log1p(Ss / Bs) - Ss
    z2 = 2.0 * f
    return np.where(ok & (z2 > 0.0), z2, NEG)


def optimal_binning(psig, wsig, wbkg, n_bins, edge_width):
    """Exact DP for the best n_bins contiguous partition maximising sum Z_i^2.

    Returns (edges_values, bins) where bins is a list of dicts with the chosen
    interval and its S, B, Z. Everything below the first chosen edge is the
    discarded background-dominated control region.
    """
    edges = np.round(np.arange(0.0, 1.0 + edge_width / 2.0, edge_width), 6)
    edges[0] = 0.0
    edges[-1] = 1.0
    edges = np.unique(edges)
    M = edges.size - 1  # bins between consecutive edges; edge indices 0..M

    hS, _ = np.histogram(psig, bins=edges, weights=wsig)
    hB, _ = np.histogram(psig, bins=edges, weights=wbkg)
    cumS = np.r_[0.0, np.cumsum(hS)]  # cumS[k] = signal below edges[k]
    cumB = np.r_[0.0, np.cumsum(hB)]

    # dp[j, k] = best sum Z^2 covering [edges[j], 1.0] with exactly k bins.
    dp = np.full((M + 1, n_bins + 1), NEG)
    choice = np.full((M + 1, n_bins + 1), -1, dtype=int)
    dp[M, 0] = 0.0  # zero bins is only valid sitting at the top edge (=1.0)

    for k in range(1, n_bins + 1):
        prev = dp[:, k - 1]
        for j in range(M - 1, -1, -1):
            jp = np.arange(j + 1, M + 1)
            S = cumS[jp] - cumS[j]
            B = cumB[jp] - cumB[j]
            z2 = _z2(S, B)
            tot = z2 + prev[jp]
            feasible = (z2 > NEG / 2) & (prev[jp] > NEG / 2)
            if not np.any(feasible):
                continue
            tot = np.where(feasible, tot, NEG)
            best_local = int(np.argmax(tot))
            if tot[best_local] > NEG / 2:
                dp[j, k] = tot[best_local]
                choice[j, k] = int(jp[best_local])

    start = int(np.argmax(dp[:, n_bins]))
    if dp[start, n_bins] <= NEG / 2:
        raise RuntimeError(
            f"No feasible {n_bins}-bin partition (try lowering min_bkg_weight "
            f"or n_signal_regions)"
        )

    # Reconstruct the chosen edges.
    idxs = [start]
    j, k = start, n_bins
    while k > 0:
        jp = int(choice[j, k])
        idxs.append(jp)
        j, k = jp, k - 1

    bins = []
    for i in range(n_bins):
        a, b = idxs[i], idxs[i + 1]
        lo, hi = float(edges[a]), float(edges[b])
        S = float(cumS[b] - cumS[a])
        B = float(cumB[b] - cumB[a])
        Z = srh.calc_Z_val(S, B)
        bins.append({"lo": lo, "hi": hi, "S": S, "B": B, "Z": Z})
    return edges, bins, float(edges[start])


def main():
    proba, y, w, _labels, _feats = srh.prepare_inputs()

    is_sig = np.isin(y, sr.SIGNAL_CLASS_INDICES)
    is_bkg = np.isin(y, sr.BACKGROUND_CLASS_INDICES)
    wsig = np.where(is_sig, w, 0.0)
    wbkg = np.where(is_bkg, w, 0.0)
    psig = proba[:, SIGNAL_AXES].sum(axis=1)

    S_total = float(wsig.sum())
    B_total = float(wbkg.sum())
    log_message(
        f"  Discriminant: p(signal) = " + " + ".join(f"p({n})" for n in SIGNAL_AXIS_NAMES)
    )
    log_message(f"  S_total={S_total:.4g}, B_total={B_total:.4g}, "
                f"edge_width={EDGE_W}, n_bins={N_BINS}, min_bkg_weight={MIN_BKG}")

    edges, bins, start_edge = optimal_binning(psig, wsig, wbkg, N_BINS, EDGE_W)

    # Per-bin stats with errors (recomputed from events for the error bars).
    rows = []
    z2_sum = 0.0
    for k, b in enumerate(bins):
        m = (psig >= b["lo"]) & ((psig < b["hi"]) if b["hi"] < 1.0 - 1e-12 else (psig >= b["lo"]))
        wS = w[m & is_sig]
        wB = w[m & is_bkg]
        S = float(wS.sum()); B = float(wB.sum())
        sS = float(np.sqrt((wS ** 2).sum())); sB = float(np.sqrt((wB ** 2).sum()))
        Z, sZ = srh.calc_Z(S, B, sS, sB)
        z2_sum += Z * Z
        rows.append({
            "bin_index": k + 1, "lo": b["lo"], "hi": b["hi"],
            "S": S, "S_err": sS, "S_entries": int((m & is_sig).sum()),
            "B": B, "B_err": sB, "B_entries": int((m & is_bkg).sum()),
            "Z": Z, "Z_err": sZ,
        })
    Z_comb = float(np.sqrt(z2_sum))

    # Report.
    log_message("")
    log_message(f"  Discarded control region: p(signal) < {start_edge:.4f} "
                f"(signal there = {float(wsig[psig < start_edge].sum()):.4g}, "
                f"{100.0 * float(wsig[psig < start_edge].sum()) / S_total:.1f}% of S)")
    log_message("  -- Optimal 1-D bins (p(signal)) --")
    header = f"  {'bin':>3} {'p(signal) range':>22} {'Z':>8} {'S':>10} {'B':>12} {'S/sqrt(B)':>10}"
    log_message(header)
    for r in rows:
        sb = r["S"] / np.sqrt(r["B"]) if r["B"] > 0 else 0.0
        rng = f"[{r['lo']:.3f}, {r['hi']:.3f})"
        log_message(f"  {r['bin_index']:>3} {rng:>22} {r['Z']:>8.4f} "
                    f"{r['S']:>10.4g} {r['B']:>12.6g} {sb:>10.4f}")
    log_message(f"  Combined significance Z_comb = sqrt(sum Z_i^2) = {Z_comb:.4f}")

    # CSV.
    import csv
    csv_path = os.path.join(sr.OUTPUT_DIR, "signal_region_1d.csv")
    with open(csv_path, "w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["bin_index", "psignal_low", "psignal_high",
                      "significance", "significance_error",
                      "S", "S_err", "S_entries", "B", "B_err", "B_entries"])
        for r in rows:
            wtr.writerow([r["bin_index"], r["lo"], r["hi"], r["Z"], r["Z_err"],
                          r["S"], r["S_err"], r["S_entries"],
                          r["B"], r["B_err"], r["B_entries"]])
    log_message(f"  Wrote {csv_path}")
    log_message(f"Finished signal_region_1d.py for tree={sr.TREE_NAME}")


if __name__ == "__main__":
    main()
