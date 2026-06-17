"""2-D optimal-binning prototype: p(signal) bands x p(QCD_total) veto.

Extends signal_region_1d.py. The discriminants are

    p(signal)     = sum of signal-class probs        (p(VVV) + p(VH))
    p(QCD_total)  = sum of the QCD-class probs        (p(QCD_LOW)+p(MID)+p(HIGH))

The signal region set is N contiguous bins in p(signal); EACH bin additionally
applies its own optimal upper veto p(QCD_total) < c. Because the bins are
disjoint in p(signal), they remain mutually non-overlapping whatever veto each
chooses, so the combined significance sum_i Z_i^2 is still maximised by an exact
dynamic program over the p(signal) edges -- the only change vs the 1-D tool is
that each candidate p(signal) interval is scored with its best QCD veto, found
by a vectorised sweep over a 2-D prefix-sum table.

This collapses the three QCD axes to a single veto axis (justified by the QCD
score-correlation analysis: one QCD class dominates per event, so p(QCD_total)
captures the QCD-ness) and reuses all data loading from signal_region_hist.

    SR_HIST_CONFIG_PATH=$PWD/config_hist.json python3 ./signal_region_2d.py
"""

import os
import numpy as np

import signal_region_hist as srh  # sets SCAN_CONFIG_PATH and imports signal_region

sr = srh.sr
log_message = sr.log_message

# -------------------- Config --------------------
_cfg = sr._load_json(srh.HIST_CFG_PATH)
N_BINS    = max(1, int(_cfg.get("n_signal_regions", 4)))
MIN_BKG   = float(sr.MIN_BKG_WEIGHT)
PSIG_W    = float(_cfg.get("onedim_bin_width", 0.005))
QCD_W     = float(_cfg.get("qcd_bin_width", _cfg.get("onedim_bin_width", 0.005)))
NEG = -1.0e18

_sig_cls = _cfg.get("signal_score_classes")
if _sig_cls is None:
    SIGNAL_AXES = list(sr.SIGNAL_CLASS_INDICES)
else:
    SIGNAL_AXES = [sr.CLASS_NAMES.index(c) if isinstance(c, str) else int(c)
                   for c in _sig_cls]
QCD_AXES = [i for i, n in enumerate(sr.CLASS_NAMES) if n.upper().startswith("QCD")]
SIGNAL_AXIS_NAMES = [sr.CLASS_NAMES[i] for i in SIGNAL_AXES]
QCD_AXIS_NAMES = [sr.CLASS_NAMES[i] for i in QCD_AXES]


def _z2(S, B):
    """Asymptotic Z^2 = 2[(S+B)ln(1+S/B)-S], vectorised; invalid -> NEG."""
    ok = (B >= MIN_BKG) & (S > 0.0)
    Bs = np.where(ok, B, 1.0)
    Ss = np.where(ok, S, 0.0)
    z2 = 2.0 * ((Ss + Bs) * np.log1p(Ss / Bs) - Ss)
    return np.where(ok & (z2 > 0.0), z2, NEG)


def optimal_2d_binning(psig, pqcd, wsig, wbkg, n_bins, psig_w, qcd_w):
    """Best n_bins p(signal) partition, each bin with its own p(QCD) veto.

    Returns list of bins {lo, hi, qcut, S, B, Z} plus the discarded start edge.
    """
    pe = np.unique(np.r_[np.round(np.arange(0.0, 1.0 + psig_w / 2.0, psig_w), 6), 1.0])
    pe[0] = 0.0; pe[-1] = 1.0
    qe = np.unique(np.r_[np.round(np.arange(0.0, 1.0 + qcd_w / 2.0, qcd_w), 6), 1.0])
    qe[0] = 0.0; qe[-1] = 1.0
    M = pe.size - 1   # p(signal) bins; edge indices 0..M
    Q = qe.size - 1   # p(QCD) bins;    edge indices 0..Q

    HSig, _, _ = np.histogram2d(psig, pqcd, bins=[pe, qe], weights=wsig)  # (M, Q)
    HBkg, _, _ = np.histogram2d(psig, pqcd, bins=[pe, qe], weights=wbkg)
    # 2-D prefix sums with leading zero row/col: H[i,j] = sum over psig_bin<i, qcd_bin<j.
    H_S = np.zeros((M + 1, Q + 1)); H_S[1:, 1:] = np.cumsum(np.cumsum(HSig, 0), 1)
    H_B = np.zeros((M + 1, Q + 1)); H_B[1:, 1:] = np.cumsum(np.cumsum(HBkg, 0), 1)

    # value[a,b] = best Z^2 of {p(signal) in [pe[a],pe[b])} with optimal QCD veto;
    # qci[a,b] = qcd edge index of that veto (region keeps p(QCD) < qe[qci]).
    value = np.full((M + 1, M + 1), NEG)
    qci = np.zeros((M + 1, M + 1), dtype=int)
    for a in range(M):
        # S,B for all b>a (rows) and all qcd cuts c (cols), with p(signal) slice [a,b).
        Sb = H_S[a + 1:] - H_S[a][None, :]   # (M-a, Q+1)
        Bb = H_B[a + 1:] - H_B[a][None, :]
        z2 = _z2(Sb, Bb)                      # (M-a, Q+1)
        c_best = np.argmax(z2, axis=1)
        v_best = z2[np.arange(z2.shape[0]), c_best]
        value[a, a + 1:] = v_best
        qci[a, a + 1:] = c_best

    # DP over p(signal) edges (same structure as the 1-D tool).
    dp = np.full((M + 1, n_bins + 1), NEG)
    choice = np.full((M + 1, n_bins + 1), -1, dtype=int)
    dp[M, 0] = 0.0
    for k in range(1, n_bins + 1):
        prev = dp[:, k - 1]
        for j in range(M - 1, -1, -1):
            cand = value[j, j + 1:] + prev[j + 1:]
            feasible = (value[j, j + 1:] > NEG / 2) & (prev[j + 1:] > NEG / 2)
            if not np.any(feasible):
                continue
            cand = np.where(feasible, cand, NEG)
            bl = int(np.argmax(cand))
            if cand[bl] > NEG / 2:
                dp[j, k] = cand[bl]
                choice[j, k] = j + 1 + bl

    start = int(np.argmax(dp[:, n_bins]))
    if dp[start, n_bins] <= NEG / 2:
        raise RuntimeError("No feasible 2-D partition (lower min_bkg_weight or n_signal_regions)")

    idxs = [start]; j, k = start, n_bins
    while k > 0:
        jp = int(choice[j, k]); idxs.append(jp); j, k = jp, k - 1

    bins = []
    for i in range(n_bins):
        a, b = idxs[i], idxs[i + 1]
        c = int(qci[a, b])
        lo, hi, qcut = float(pe[a]), float(pe[b]), float(qe[c])
        S = float(H_S[b, c] - H_S[a, c]); B = float(H_B[b, c] - H_B[a, c])
        bins.append({"lo": lo, "hi": hi, "qcut": qcut, "S": S, "B": B,
                     "Z": srh.calc_Z_val(S, B)})
    return bins, float(pe[start])


def main():
    proba, y, w, _labels, _feats = srh.prepare_inputs()
    is_sig = np.isin(y, sr.SIGNAL_CLASS_INDICES)
    is_bkg = np.isin(y, sr.BACKGROUND_CLASS_INDICES)
    wsig = np.where(is_sig, w, 0.0)
    wbkg = np.where(is_bkg, w, 0.0)
    psig = proba[:, SIGNAL_AXES].sum(axis=1)
    pqcd = proba[:, QCD_AXES].sum(axis=1)
    S_total = float(wsig.sum()); B_total = float(wbkg.sum())

    log_message("  Discriminants: p(signal) = " + " + ".join(f"p({n})" for n in SIGNAL_AXIS_NAMES)
                + " ;  veto p(QCD_total) = " + " + ".join(f"p({n})" for n in QCD_AXIS_NAMES))
    log_message(f"  S_total={S_total:.4g}, B_total={B_total:.4g}, psig_w={PSIG_W}, "
                f"qcd_w={QCD_W}, n_bins={N_BINS}, min_bkg_weight={MIN_BKG}")

    bins, start_edge = optimal_2d_binning(psig, pqcd, wsig, wbkg, N_BINS, PSIG_W, QCD_W)

    rows = []; z2_sum = 0.0
    for k, b in enumerate(bins):
        m = (psig >= b["lo"])
        m &= (psig < b["hi"]) if b["hi"] < 1.0 - 1e-12 else (psig <= 1.0)
        if b["qcut"] < 1.0 - 1e-12:
            m &= (pqcd < b["qcut"])
        wS = w[m & is_sig]; wB = w[m & is_bkg]
        S = float(wS.sum()); B = float(wB.sum())
        sS = float(np.sqrt((wS ** 2).sum())); sB = float(np.sqrt((wB ** 2).sum()))
        Z, sZ = srh.calc_Z(S, B, sS, sB)
        z2_sum += Z * Z
        rows.append({"bin_index": k + 1, "lo": b["lo"], "hi": b["hi"], "qcut": b["qcut"],
                     "S": S, "S_err": sS, "S_entries": int((m & is_sig).sum()),
                     "B": B, "B_err": sB, "B_entries": int((m & is_bkg).sum()),
                     "Z": Z, "Z_err": sZ})
    Z_comb = float(np.sqrt(z2_sum))

    log_message("")
    log_message(f"  Discarded control region: p(signal) < {start_edge:.4f}")
    log_message("  -- Optimal 2-D bins (p(signal) band x p(QCD_total) veto) --")
    log_message(f"  {'bin':>3} {'p(signal) range':>20} {'p(QCD)<':>9} {'Z':>8} "
                f"{'S':>9} {'B':>12} {'S/sqrt(B)':>10}")
    for r in rows:
        sb = r["S"] / np.sqrt(r["B"]) if r["B"] > 0 else 0.0
        qc = f"{r['qcut']:.3f}" if r["qcut"] < 1.0 - 1e-12 else "(none)"
        rng = f"[{r['lo']:.3f}, {r['hi']:.3f})"
        log_message(f"  {r['bin_index']:>3} {rng:>20} {qc:>9} {r['Z']:>8.4f} "
                    f"{r['S']:>9.4g} {r['B']:>12.6g} {sb:>10.4f}")
    log_message(f"  Combined significance Z_comb = sqrt(sum Z_i^2) = {Z_comb:.4f}")

    import csv
    csv_path = os.path.join(sr.OUTPUT_DIR, "signal_region_2d.csv")
    with open(csv_path, "w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["bin_index", "psignal_low", "psignal_high", "pqcd_veto_high",
                      "significance", "significance_error",
                      "S", "S_err", "S_entries", "B", "B_err", "B_entries"])
        for r in rows:
            wtr.writerow([r["bin_index"], r["lo"], r["hi"], r["qcut"], r["Z"], r["Z_err"],
                          r["S"], r["S_err"], r["S_entries"], r["B"], r["B_err"], r["B_entries"]])
    log_message(f"  Wrote {csv_path}")
    log_message(f"Finished signal_region_2d.py for tree={sr.TREE_NAME}")


if __name__ == "__main__":
    main()
