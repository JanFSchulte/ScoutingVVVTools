"""3-D optimal-binning prototype: p(signal) bands x p(QCD) veto x 3rd-axis veto.

Extends signal_region_2d.py with one more per-bin veto axis. Because the softmax
probabilities satisfy p(signal)+p(QCD_total)+p(EW)=1, the *collapsed* EW score
adds no genuine information on top of p(signal) and p(QCD_total); a real third
dimension must be an INDIVIDUAL background class (Top / VT / VV), which is not
determined by the collapsed sums. This script loads the data once and evaluates
the exact 3-D DP for several candidate third axes, reporting which one helps most
and by how much vs the 2-D result.

Regions: {p(signal) in [a,b)} ∩ {p(QCD_total) < cq} ∩ {p(B3) < c3}, with the
p(signal) bands contiguous (so disjoint) and each bin choosing its own (cq, c3).
The veto grids are coarser (default 0.02) to keep the 2-D veto sweep tractable.

    SR_HIST_CONFIG_PATH=$PWD/config_hist.json python3 ./signal_region_3d.py
"""

import os
import numpy as np

import signal_region_hist as srh

sr = srh.sr
log_message = sr.log_message

_cfg = sr._load_json(srh.HIST_CFG_PATH)
N_BINS   = max(1, int(_cfg.get("n_signal_regions", 4)))
MIN_BKG  = float(sr.MIN_BKG_WEIGHT)
PSIG_W   = float(_cfg.get("onedim_bin_width", 0.005))
VETO_W   = float(_cfg.get("veto_bin_width", 0.02))
NEG = -1.0e18

SIGNAL_AXES = list(sr.SIGNAL_CLASS_INDICES)
QCD_AXES = [i for i, n in enumerate(sr.CLASS_NAMES) if n.upper().startswith("QCD")]
# Candidate third veto axes: individual EW backgrounds, plus the (redundant) EW sum
# as a control to demonstrate it does not help.
EW_AXES = [i for i in sr.BACKGROUND_CLASS_INDICES if i not in QCD_AXES]
THIRD_CANDIDATES = _cfg.get("third_axis_candidates")  # optional override (list of names)


def _z2(S, B):
    ok = (B >= MIN_BKG) & (S > 0.0)
    Bs = np.where(ok, B, 1.0)
    Ss = np.where(ok, S, 0.0)
    z2 = 2.0 * ((Ss + Bs) * np.log1p(Ss / Bs) - Ss)
    return np.where(ok & (z2 > 0.0), z2, NEG)


def _grid(w):
    e = np.unique(np.r_[np.round(np.arange(0.0, 1.0 + w / 2.0, w), 6), 1.0])
    e[0] = 0.0; e[-1] = 1.0
    return e


def value_table(psig, pqcd, pb3, wsig, wbkg, pe, qe, te):
    """value[a,b] = best Z^2 of p(signal) band [a,b) over (qcd, b3) vetoes."""
    M, Q, T = pe.size - 1, qe.size - 1, te.size - 1
    HS, _ = np.histogramdd(np.column_stack([psig, pqcd, pb3]), bins=[pe, qe, te], weights=wsig)
    HB, _ = np.histogramdd(np.column_stack([psig, pqcd, pb3]), bins=[pe, qe, te], weights=wbkg)
    H_S = np.zeros((M + 1, Q + 1, T + 1))
    H_B = np.zeros((M + 1, Q + 1, T + 1))
    H_S[1:, 1:, 1:] = np.cumsum(np.cumsum(np.cumsum(HS, 0), 1), 2)
    H_B[1:, 1:, 1:] = np.cumsum(np.cumsum(np.cumsum(HB, 0), 1), 2)

    value = np.full((M + 1, M + 1), NEG)
    cqi = np.zeros((M + 1, M + 1), dtype=int)
    cti = np.zeros((M + 1, M + 1), dtype=int)
    for a in range(M):
        Sb = H_S[a + 1:] - H_S[a][None]   # (M-a, Q+1, T+1)
        Bb = H_B[a + 1:] - H_B[a][None]
        z2 = _z2(Sb, Bb)
        flat = z2.reshape(z2.shape[0], -1)
        bi = np.argmax(flat, axis=1)
        bv = flat[np.arange(flat.shape[0]), bi]
        cq, ct = np.unravel_index(bi, (Q + 1, T + 1))
        value[a, a + 1:] = bv
        cqi[a, a + 1:] = cq
        cti[a, a + 1:] = ct
    return value, cqi, cti, H_S, H_B


def dp_partition(value, n_bins):
    M = value.shape[0] - 1
    dp = np.full((M + 1, n_bins + 1), NEG)
    choice = np.full((M + 1, n_bins + 1), -1, dtype=int)
    dp[M, 0] = 0.0
    for k in range(1, n_bins + 1):
        prev = dp[:, k - 1]
        for j in range(M - 1, -1, -1):
            cand = value[j, j + 1:] + prev[j + 1:]
            feas = (value[j, j + 1:] > NEG / 2) & (prev[j + 1:] > NEG / 2)
            if not np.any(feas):
                continue
            cand = np.where(feas, cand, NEG)
            bl = int(np.argmax(cand))
            if cand[bl] > NEG / 2:
                dp[j, k] = cand[bl]
                choice[j, k] = j + 1 + bl
    start = int(np.argmax(dp[:, n_bins]))
    if dp[start, n_bins] <= NEG / 2:
        return None, None
    idxs = [start]; j, k = start, n_bins
    while k > 0:
        jp = int(choice[j, k]); idxs.append(jp); j, k = jp, k - 1
    return idxs, float(dp[start, n_bins])


def main():
    proba, y, w, _l, _f = srh.prepare_inputs()
    is_sig = np.isin(y, sr.SIGNAL_CLASS_INDICES)
    is_bkg = np.isin(y, sr.BACKGROUND_CLASS_INDICES)
    wsig = np.where(is_sig, w, 0.0); wbkg = np.where(is_bkg, w, 0.0)
    psig = proba[:, SIGNAL_AXES].sum(axis=1)
    pqcd = proba[:, QCD_AXES].sum(axis=1)

    log_message(f"  S_total={wsig.sum():.4g}, B_total={wbkg.sum():.4g}, psig_w={PSIG_W}, "
                f"veto_w={VETO_W}, n_bins={N_BINS}")
    log_message(f"  Base 2-D = p(signal) x p(QCD_total). Testing 3rd veto axes...")

    pe, qe, te = _grid(PSIG_W), _grid(VETO_W), _grid(VETO_W)

    if THIRD_CANDIDATES:
        cand_axes = [(sr.CLASS_NAMES.index(c), c) for c in THIRD_CANDIDATES]
    else:
        cand_axes = [(i, sr.CLASS_NAMES[i]) for i in EW_AXES]
        cand_axes.append((None, "EW_sum(Top+VT+VV) [redundant control]"))

    results = []
    for axis, name in cand_axes:
        pb3 = (proba[:, EW_AXES].sum(axis=1) if axis is None else proba[:, axis])
        value, cqi, cti, H_S, H_B = value_table(psig, pqcd, pb3, wsig, wbkg, pe, qe, te)
        idxs, sumz2 = dp_partition(value, N_BINS)
        if idxs is None:
            log_message(f"    3rd axis p({name}): infeasible")
            continue
        zc = float(np.sqrt(sumz2))
        results.append((zc, name, idxs, cqi, cti, pb3))
        log_message(f"    3rd axis p({name:40s}: Z_comb = {zc:.4f}")

    results.sort(key=lambda r: -r[0])
    zc, name, idxs, cqi, cti, pb3 = results[0]
    log_message("")
    log_message(f"  Best 3rd axis: p({name})  ->  Z_comb = {zc:.4f}")
    log_message("  -- Optimal 3-D bins --")
    log_message(f"  {'bin':>3} {'p(signal) range':>20} {'p(QCD)<':>9} {'3rd<':>9} "
                f"{'Z':>8} {'S':>9} {'B':>12}")
    rows = []
    for i in range(N_BINS):
        a, b = idxs[i], idxs[i + 1]
        cq, ct = int(cqi[a, b]), int(cti[a, b])
        lo, hi, qc, tc = float(pe[a]), float(pe[b]), float(qe[cq]), float(te[ct])
        m = (psig >= lo)
        m &= (psig < hi) if hi < 1 - 1e-12 else (psig <= 1.0)
        if qc < 1 - 1e-12: m &= (pqcd < qc)
        if tc < 1 - 1e-12: m &= (pb3 < tc)
        wS = w[m & is_sig]; wB = w[m & is_bkg]
        S = float(wS.sum()); B = float(wB.sum())
        sS = float(np.sqrt((wS ** 2).sum())); sB = float(np.sqrt((wB ** 2).sum()))
        Z, sZ = srh.calc_Z(S, B, sS, sB)
        rows.append((i + 1, lo, hi, qc, tc, Z, sZ, S, sS, B, sB,
                     int((m & is_sig).sum()), int((m & is_bkg).sum())))
        qcs = f"{qc:.3f}" if qc < 1 - 1e-12 else "(none)"
        tcs = f"{tc:.3f}" if tc < 1 - 1e-12 else "(none)"
        log_message(f"  {i+1:>3} [{lo:.3f}, {hi:.3f}){'':>4} {qcs:>9} {tcs:>9} "
                    f"{Z:>8.4f} {S:>9.4g} {B:>12.6g}")
    log_message(f"  Combined Z = {float(np.sqrt(sum(r[5]**2 for r in rows))):.4f}")

    import csv
    csv_path = os.path.join(sr.OUTPUT_DIR, "signal_region_3d.csv")
    with open(csv_path, "w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["bin_index", "psignal_low", "psignal_high", "pqcd_veto_high",
                      "third_veto_high", "third_axis",
                      "significance", "significance_error",
                      "S", "S_err", "S_entries", "B", "B_err", "B_entries"])
        for r in rows:
            wtr.writerow([r[0], r[1], r[2], r[3], r[4], name, r[5], r[6],
                          r[7], r[8], r[11], r[9], r[10], r[12]])
    log_message(f"  Wrote {csv_path}")
    log_message(f"Finished signal_region_3d.py for tree={sr.TREE_NAME}")


if __name__ == "__main__":
    main()
