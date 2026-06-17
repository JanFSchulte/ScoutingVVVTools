"""Production signal-region tool: exact optimal binning of derived discriminants.

Defines the signal regions as N contiguous bins of

    p(signal) = sum of signal-class probabilities          (p(VVV)+p(VH))

each with its own optimal vetoes on

    p(QCD_total) = sum of the QCD-class probabilities       (collapses the 3 QCD
                                                             classes to one axis)
    p(<3rd>)     = the single individual background class that helps most,
                   auto-selected from the EW backgrounds (Top/VT/VV).

The optimal partition (maximising the combined significance sum_i Z_i^2, each bin
holding >= min_bkg_weight background) is found EXACTLY by dynamic programming --
no beam search / branch-and-bound. It dominates the multidimensional box search on
this dataset while being instant, globally optimal for its region family, and
trivially fit-ready.

Output mirrors the original signal_region.py: the same per-bin text report with
per-class breakdowns, the same signal_region.csv schema (the three discriminants
play the role of the scan axes), the per-class score-distribution plots and the
multiclass simplex scatter, plus dedicated binning plots. Everything is written to
a sibling "<output_dir>_binned" directory so it never clobbers the box-search
outputs; point the downstream combine step at that directory for a drop-in swap.

    SR_HIST_CONFIG_PATH=$PWD/config_hist.json python3 ./signal_region_binned.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import signal_region_hist as srh
import signal_region_3d as s3

sr = srh.sr
log_message = sr.log_message
log_warning = sr.log_warning

# -------------------- Config --------------------
_cfg = sr._load_json(srh.HIST_CFG_PATH)
N_BINS    = max(1, int(_cfg.get("n_signal_regions", 4)))
MIN_BKG   = float(sr.MIN_BKG_WEIGHT)
PSIG_W    = float(_cfg.get("onedim_bin_width", 0.005))
VETO_W    = float(_cfg.get("veto_bin_width", 0.01))      # final (fine) veto grid
SCAN_W    = float(_cfg.get("veto_scan_width", 0.02))     # coarse grid for axis pick
SIGNAL_AXES = list(sr.SIGNAL_CLASS_INDICES)
QCD_AXES = [i for i, n in enumerate(sr.CLASS_NAMES) if n.upper().startswith("QCD")]
EW_AXES = [i for i in sr.BACKGROUND_CLASS_INDICES if i not in QCD_AXES]
_third = _cfg.get("third_axis_candidates")
if _third:
    THIRD_CANDS = [sr.CLASS_NAMES.index(c) if isinstance(c, str) else int(c) for c in _third]
else:
    THIRD_CANDS = list(EW_AXES)

BINNED_OUT = sr.OUTPUT_DIR.rstrip("/").rstrip("\\") + "_binned"


# -------------------- Region -> top_bins (reuses sr reporting) --------------------
def build_top_bins(psig, pqcd, pb3, y, w, is_sig, is_bkg, bins_def, third_name,
                   S_total, B_total):
    """Construct the original top_bins schema with the 3 discriminants as axes."""
    axis_names = ["p_signal", "p_QCD_total", f"p_{third_name}"]
    top_bins = []
    for k, (lo, hi, qcut, tcut) in enumerate(bins_def):
        m = (psig >= lo)
        m &= (psig < hi) if hi < 1.0 - 1e-12 else (psig <= 1.0)
        if qcut < 1.0 - 1e-12:
            m &= (pqcd < qcut)
        if tcut < 1.0 - 1e-12:
            m &= (pb3 < tcut)

        wS = w[m & is_sig]; wB = w[m & is_bkg]
        S = float(wS.sum()); B = float(wB.sum())
        sS = float(np.sqrt((wS ** 2).sum())); sB = float(np.sqrt((wB ** 2).sum()))
        S_e = int((m & is_sig).sum()); B_e = int((m & is_bkg).sum())
        Z, sZ = srh.calc_Z(S, B, sS, sB)

        W_bin = S + B; w2_bin = sS ** 2 + sB ** 2
        cats = []
        for ci, cn in enumerate(sr.CLASS_NAMES):
            mc = (y == ci) & m; wc = w[mc]
            Sj = float(wc.sum()); sSj = float(np.sqrt((wc ** 2).sum()))
            Bj = W_bin - Sj; sBj = float(np.sqrt(max(0.0, w2_bin - sSj ** 2)))
            Zj, sZj = srh.calc_Z(Sj, Bj, sSj, sBj)
            cats.append({"name": cn, "S": Sj, "S_err": sSj, "B": Bj, "B_err": sBj,
                         "Z": Zj, "Z_err": sZj})
        bkgs = []
        for bi in sr.BACKGROUND_CLASS_INDICES:
            mc = (y == bi) & m; wc = w[mc]
            bkgs.append({"name": sr.CLASS_NAMES[bi], "B": float(wc.sum()),
                         "B_err": float(np.sqrt((wc ** 2).sum()))})

        top_bins.append({
            "bin_index": k + 1,
            "thr_low": np.array([lo, 0.0, 0.0]),
            "thr_high": np.array([hi, qcut, tcut]),
            "axis_names": list(axis_names),
            "significance": Z, "significance_error": sZ,
            "S": S, "S_err": sS, "S_entries": S_e,
            "B": B, "B_err": sB, "B_entries": B_e,
            "categories": cats, "backgrounds": bkgs,
            "bin_signal_efficiency": (S / S_total) if S_total > 0 else float("nan"),
            "bin_background_efficiency": (B / B_total) if B_total > 0 else float("nan"),
            "tail_signal_efficiency": [], "tail_background_efficiency": [],
        })
    return top_bins


# -------------------- Plots --------------------
def plot_psignal_partition(psig, wsig, wbkg, bins_def, start_edge, z_comb):
    bins = np.linspace(0.0, 1.0, 201)
    palette = sr._plot_colors(N_BINS + 2)
    plt.figure(figsize=(9, 6))
    # Normalised shapes so signal (tiny yield) and background are comparable.
    for arr, lab, col in [(wbkg, "Background", "0.5"), (wsig, "Signal", palette[0])]:
        h, _ = np.histogram(psig, bins=bins, weights=arr)
        tot = h.sum()
        if tot > 0:
            plt.step(bins[:-1], h / tot, where="post", color=col, linewidth=2, label=lab)
    ymax = plt.ylim()[1]
    if start_edge > 1e-6:
        plt.axvspan(0.0, start_edge, color="0.85", alpha=0.5, label="control (discarded)")
    for i, (lo, hi, qc, tc) in enumerate(bins_def):
        plt.axvline(lo, color=palette[i + 1], linestyle="--", linewidth=1.4)
        plt.axvline(hi, color=palette[i + 1], linestyle="--", linewidth=1.4)
        plt.text(0.5 * (lo + hi), ymax * 0.5, f"SR{i+1}", ha="center", va="center",
                 color=palette[i + 1], fontsize=11, rotation=90)
    plt.yscale("log")
    plt.xlim(0, 1)
    plt.xlabel(r"$p(\mathrm{signal}) = p(\mathrm{VVV}) + p(\mathrm{VH})$")
    plt.ylabel("Normalised events")
    plt.title(f"Optimal {N_BINS}-bin partition   $Z_{{comb}} = {z_comb:.3f}$", fontsize=13)
    plt.legend(fontsize=11)
    sr._savefig("sr_binning_psignal")


def plot_plane(psig, pother, wsig, wbkg, is_sig, bins_def, cut_index, other_label, stem):
    """2-D background density in (p(signal), p(other)) with each bin's box drawn."""
    palette = sr._plot_colors(N_BINS + 2)
    xedges = np.linspace(0, 1, 101); yedges = np.linspace(0, 1, 101)
    Hb, _, _ = np.histogram2d(psig, pother, bins=[xedges, yedges], weights=wbkg)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    Hb_m = np.ma.masked_where(Hb <= 0, Hb)
    from matplotlib.colors import LogNorm
    pcm = ax.pcolormesh(xedges, yedges, Hb_m.T, norm=LogNorm(), cmap="Greys")
    fig.colorbar(pcm, ax=ax, label="Background (weighted)")
    # signal scatter (subsample)
    sidx = np.flatnonzero(is_sig)
    if sidx.size > 4000:
        sidx = sidx[np.linspace(0, sidx.size - 1, 4000).astype(int)]
    ax.scatter(psig[sidx], pother[sidx], s=4, alpha=0.25, color=palette[0],
               edgecolors="none", label="Signal", rasterized=True)
    for i, bd in enumerate(bins_def):
        lo, hi = bd[0], bd[1]
        cut = bd[cut_index]
        top = cut if cut < 1.0 - 1e-12 else 1.0
        ax.add_patch(Rectangle((lo, 0.0), hi - lo, top, fill=False,
                               edgecolor=palette[i + 1], linewidth=2.0,
                               label=f"SR{i+1}"))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel(r"$p(\mathrm{signal})$")
    ax.set_ylabel(other_label)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.95)
    sr._savefig(stem)


# -------------------- Main --------------------
def main():
    os.makedirs(BINNED_OUT, exist_ok=True)
    sr.OUTPUT_DIR = BINNED_OUT  # redirect reused sr writers/plotters here

    proba, y, w, _labels, _feats = srh.prepare_inputs()
    is_sig = np.isin(y, sr.SIGNAL_CLASS_INDICES)
    is_bkg = np.isin(y, sr.BACKGROUND_CLASS_INDICES)
    wsig = np.where(is_sig, w, 0.0); wbkg = np.where(is_bkg, w, 0.0)
    psig = proba[:, SIGNAL_AXES].sum(axis=1)
    pqcd = proba[:, QCD_AXES].sum(axis=1)
    S_total = float(wsig.sum()); B_total = float(wbkg.sum())

    log_message(f"Running signal_region_binned.py: tree={sr.TREE_NAME}, "
                f"n_signal_regions={N_BINS}, output_dir={BINNED_OUT}")
    log_message(f"  S_total={S_total:.4g}, B_total={B_total:.4g}, "
                f"psig_w={PSIG_W}, scan_w={SCAN_W}, veto_w={VETO_W}")
    log_message(f"  Discriminants: p(signal)=p(VVV)+p(VH); "
                f"p(QCD_total)=sum of {len(QCD_AXES)} QCD classes (collapsed)")

    # ---- Auto-select the 3rd veto axis on a coarse grid. ----
    pe_s, qe_s, te_s = s3._grid(PSIG_W), s3._grid(SCAN_W), s3._grid(SCAN_W)
    scan = []
    for ax in THIRD_CANDS:
        pb3 = proba[:, ax]
        value, _cq, _ct, _hs, _hb = s3.value_table(psig, pqcd, pb3, wsig, wbkg, pe_s, qe_s, te_s)
        idxs, sumz2 = s3.dp_partition(value, N_BINS)
        if idxs is None:
            continue
        scan.append((float(np.sqrt(sumz2)), ax, sr.CLASS_NAMES[ax]))
        log_message(f"  3rd-axis candidate p({sr.CLASS_NAMES[ax]}): Z_comb={np.sqrt(sumz2):.4f} (coarse)")
    if not scan:
        raise RuntimeError("No feasible binning for any 3rd-axis candidate")
    scan.sort(key=lambda r: -r[0])
    best_ax, best_name = scan[0][1], scan[0][2]
    runner = f"{scan[1][2]} ({scan[1][0]:.4f})" if len(scan) > 1 else "n/a"
    log_message(f"  Selected 3rd axis: p({best_name}) "
                f"(coarse Z={scan[0][0]:.4f}; runner-up: {runner})")

    # ---- Final partition for the chosen axis on the fine grid. ----
    pb3 = proba[:, best_ax]
    pe, qe, te = s3._grid(PSIG_W), s3._grid(VETO_W), s3._grid(VETO_W)
    value, cqi, cti, H_S, H_B = s3.value_table(psig, pqcd, pb3, wsig, wbkg, pe, qe, te)
    idxs, sumz2 = s3.dp_partition(value, N_BINS)
    bins_def = []
    for i in range(N_BINS):
        a, b = idxs[i], idxs[i + 1]
        bins_def.append((float(pe[a]), float(pe[b]),
                         float(qe[int(cqi[a, b])]), float(te[int(cti[a, b])])))
    start_edge = float(pe[idxs[0]])
    log_message(f"  Discarded control region: p(signal) < {start_edge:.4f} "
                f"({100.0*float(wsig[psig < start_edge].sum())/S_total:.1f}% of S)")

    # ---- Build report-compatible result and reuse sr reporting/plots. ----
    top_bins = build_top_bins(psig, pqcd, pb3, y, w, is_sig, is_bkg, bins_def,
                              best_name, S_total, B_total)
    summary = {
        "selector": f"exact DP binning [p(signal) x p(QCD_total) x p({best_name})]",
        "completed": True, "nodes": len(pe),
        "objective_sum_z2": float(sumz2),
        "objective_upper_bound_sum_z2": float(sumz2),
        "geometry_overlap_pairs": 0, "event_overlap_pairs": 0,
        "candidate_count": len(pe),
    }
    result = sr._make_signal_region_result(top_bins, S_total, B_total, summary)

    log_message("Plotting score distributions")
    sr.plot_score_distributions(proba, y, w)
    log_message("Plotting multiclass simplex (scatter)")
    sr.plot_signal_regions_2d({"top_bins": []}, proba, y, w)
    log_message("Plotting binning")
    plot_psignal_partition(psig, wsig, wbkg, bins_def, start_edge,
                           result["combined_significance"])
    plot_plane(psig, pqcd, wsig, wbkg, is_sig, bins_def, 2,
               r"$p(\mathrm{QCD\_total})$", "sr_binning_signal_vs_qcd")
    plot_plane(psig, pb3, wsig, wbkg, is_sig, bins_def, 3,
               rf"$p(\mathrm{{{best_name}}})$", f"sr_binning_signal_vs_{best_name.lower()}")

    sr.print_results(result)
    sr.write_signal_region_csv(result)
    log_message(f"Finished signal_region_binned.py for tree={sr.TREE_NAME}")


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        log_message(f"Runtime error: {ex}")
        raise
