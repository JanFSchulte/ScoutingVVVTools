#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Data vs MC histogram comparison plotter.

Reads convert_branch.C output ROOT files directly (per-sample trees), optionally
applies the trained-model selection.json clip/threshold cuts (no log transform),
and draws a stacked MC + data panel with a Data/MC ratio sub-panel.

Pass --no-selection to skip threshold and clip-range cuts on plot variables
(BDT model clip ranges are still applied internally for score computation).
"""

import argparse
import os
import sys
import json
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
import uproot


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
_BDT_DIR = os.path.join(_ROOT_DIR, "selections", "BDT")
if _BDT_DIR not in sys.path:
    sys.path.insert(0, _BDT_DIR)

from model_io import (
    load_model as _shared_load_model,
    predict_model_proba as _shared_predict_model_proba,
)

# -------------------- Style --------------------
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["mathtext.rm"] = "serif"
plt.style.use(hep.style.CMS)


# -------------------- Helpers --------------------
def log_message(msg):
    print(msg, flush=True)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _resolve(path, base):
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(base, path))


def _score_branch_name(class_name):
    return f"score_{class_name}"


# -------------------- Config loading --------------------
_cfg_path = os.environ.get("PLOT_CONFIG_PATH", os.path.join(_SCRIPT_DIR, "config.json"))
_cfg_path = _resolve(_cfg_path, _SCRIPT_DIR)

plot_cfg             = _load_json(_cfg_path)
branch_overrides_cfg = _load_json(os.path.join(_SCRIPT_DIR, "branch.json"))

SUBMIT_TREES          = plot_cfg.get("submit_trees", ["fat2", "fat3"])
DATA_SAMPLES          = list(plot_cfg.get("data_samples", []))
DEFAULT_BINS          = int(plot_cfg.get("default_bins", 10))
OUTPUT_ROOT_PATT      = plot_cfg.get("output_root", "./pre-selection/{tree_name}")
OUTPUT_ROOT_NOSEL_PATT = plot_cfg.get("output_root_nosel", "./no-selection/{tree_name}")
BDT_ROOT_PATT         = plot_cfg["bdt_root"]

SAMPLE_CFG_PATH         = _resolve(plot_cfg["sample_config"], _SCRIPT_DIR)
CONVERT_BRANCH_CFG_PATH = _resolve(plot_cfg["convert_branch_config"], _SCRIPT_DIR)

sample_cfg         = _load_json(SAMPLE_CFG_PATH)
convert_branch_cfg = _load_json(CONVERT_BRANCH_CFG_PATH)

_CONVERT_CFG_PATH = os.path.join(os.path.dirname(CONVERT_BRANCH_CFG_PATH), "config.json")
convert_cfg       = _load_json(_CONVERT_CFG_PATH)

SAMPLE_INFO = {s["name"]: s for s in sample_cfg["sample"]}


def _compute_lumi_total():
    total = 0.0
    for name in DATA_SAMPLES:
        if name not in SAMPLE_INFO:
            raise RuntimeError(f"Data sample '{name}' not found in sample.json")
        info = SAMPLE_INFO[name]
        if info.get("is_MC", True):
            raise RuntimeError(f"Sample '{name}' is flagged as MC in sample.json but listed as data")
        total += float(info.get("lumi", 0.0))
    return total


LUMI_TOTAL = _compute_lumi_total()


# -------------------- Branch discovery --------------------
def _tree_plot_cfg(tree_name):
    if not isinstance(branch_overrides_cfg, dict):
        return {}
    tree_cfg = branch_overrides_cfg.get(tree_name, {})
    return tree_cfg if isinstance(tree_cfg, dict) else {}


def _skip_branches_for_tree(tree_name):
    skip = _tree_plot_cfg(tree_name).get("skip_branches", [])
    if not isinstance(skip, list):
        raise TypeError(f"plotting/branch.json:{tree_name}.skip_branches must be a list")
    return set(skip)


def _tree_output_entry(tree_name):
    for tree in convert_branch_cfg["output"]["trees"]:
        if tree["name"] == tree_name:
            return tree
    raise KeyError(f"Tree '{tree_name}' not in convert branch config")


def _plot_branches_for_tree(tree_name):
    """Return branch names to plot (onlyMC=false, not skipped, slots expanded)."""
    tree    = _tree_output_entry(tree_name)
    skip    = _skip_branches_for_tree(tree_name)
    scalars = tree.get("scalars", {})
    entries = list(scalars.get("regular", [])) + list(scalars.get("extrema", []))
    out, seen = [], set()
    for e in entries:
        if e.get("onlyMC", False):
            continue
        name = e["name"]
        slots = e.get("slots")
        if slots:
            for i in range(int(slots)):
                n = f"{name}_{i + 1}"
                if n in skip or n in seen:
                    continue
                seen.add(n)
                out.append(n)
        else:
            if name in skip or name in seen:
                continue
            seen.add(name)
            out.append(name)
    return out


# -------------------- Trained-model config copies --------------------
def _bdt_root_for_tree(tree_name):
    return _resolve(BDT_ROOT_PATT.format(tree_name=tree_name), _SCRIPT_DIR)


def _bdt_configs_for_tree(tree_name, load_test_meta=True):
    bdt_root = _bdt_root_for_tree(tree_name)
    cfg = _load_json(os.path.join(bdt_root, "config.json"))
    br = _load_json(os.path.join(bdt_root, "branch.json"))
    sel = _load_json(os.path.join(bdt_root, "selection.json"))
    meta = _load_json(os.path.join(bdt_root, "test_ranges.json")) if load_test_meta else None
    return cfg, br, sel, meta


# -------------------- Input file resolution --------------------
def _sample_group(info):
    if not info.get("is_MC", True):
        return "data"
    return "signal" if info.get("is_signal", False) else "bkg"


def _input_files(sample_name, input_root, input_pattern):
    info = SAMPLE_INFO[sample_name]
    sg   = _sample_group(info)
    base = input_pattern.format(input_root=input_root, sample_group=sg, sample=sample_name)
    stem = base[:-5] if base.endswith(".root") else base
    return sorted(glob.glob(base) + glob.glob(stem + "_*.root"))


def _tree_entries_total(files, tree_name):
    total = 0
    for fpath in files:
        with uproot.open(fpath) as uf:
            if tree_name not in uf:
                continue
            total += int(uf[tree_name].num_entries)
    return total


def _load_tree(files, tree_name, branches, strict=True):
    parts = []
    for fpath in files:
        with uproot.open(fpath) as uf:
            if tree_name not in uf:
                continue
            tree  = uf[tree_name]
            avail = set(tree.keys())
            missing = [b for b in branches if b not in avail]
            if missing:
                if strict:
                    raise KeyError(
                        f"Missing branches in {fpath}:{tree_name}: "
                        f"{', '.join(missing[:10])}" + (" ..." if len(missing) > 10 else "")
                    )
                log_message(
                    f"  [WARN] skipping {len(missing)} missing branch(es) in "
                    f"{os.path.basename(fpath)}:{tree_name}: "
                    f"{', '.join(missing[:5])}" + (" ..." if len(missing) > 5 else "")
                )
            load = [b for b in branches if b in avail]
            parts.append(tree.arrays(load, library="pd"))
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


_CHUNK_SIZE = "200 MB"


def _iter_chunks(files, tree_name, branches, strict=True):
    """Yield DataFrame chunks from a list of files without loading all events at once."""
    for fpath in files:
        with uproot.open(fpath) as uf:
            if tree_name not in uf:
                continue
            tree  = uf[tree_name]
            avail = set(tree.keys())
            missing = [b for b in branches if b not in avail]
            if missing:
                if strict:
                    raise KeyError(
                        f"Missing branches in {fpath}:{tree_name}: "
                        f"{', '.join(missing[:10])}" + (" ..." if len(missing) > 10 else "")
                    )
                log_message(
                    f"  [WARN] skipping {len(missing)} missing branch(es) in "
                    f"{os.path.basename(fpath)}:{tree_name}: "
                    f"{', '.join(missing[:5])}" + (" ..." if len(missing) > 5 else "")
                )
            load = [b for b in branches if b in avail]
            if not load:
                continue
            for chunk in tree.iterate(expressions=load, step_size=_CHUNK_SIZE, library="pd"):
                yield chunk


def _prepass(files, tree_name, auto_range_branches, reweight_branches, branch_logx, strict):
    """Single streaming pass to collect per-branch value ranges and the raw-weight sum.

    Returns (raw_w_sum, ranges) where ranges = {branch: (lo, hi)}.
    raw_w_sum is the sum of the product of reweight_branches over all events (1.0 each
    if reweight_branches is empty), used later to normalise MC event weights.
    """
    prepass_cols = sorted(set(auto_range_branches) | set(reweight_branches))
    raw_w_sum = 0.0
    mins = {b:  np.inf for b in auto_range_branches}
    maxs = {b: -np.inf for b in auto_range_branches}

    for chunk in _iter_chunks(files, tree_name, prepass_cols, strict=strict):
        raw_w = np.ones(len(chunk), dtype=float)
        for rb in reweight_branches:
            if rb in chunk.columns:
                raw_w *= chunk[rb].to_numpy(dtype=float, copy=False)
        raw_w_sum += float(raw_w.sum())

        for b in auto_range_branches:
            if b not in chunk.columns:
                continue
            arr = chunk[b].to_numpy(dtype=float, copy=False)
            valid = arr[arr >= -990]
            if branch_logx.get(b, False):
                valid = valid[valid > 0]
            if valid.size == 0:
                continue
            mins[b] = min(mins[b], float(valid.min()))
            maxs[b] = max(maxs[b], float(valid.max()))

    ranges = {}
    for b in auto_range_branches:
        lo, hi = mins[b], maxs[b]
        if np.isfinite(lo) and np.isfinite(hi) and lo <= hi:
            ranges[b] = (lo, hi)

    return raw_w_sum, ranges


def _stream_hists(files, tree_name, branch_edges, target_total, raw_w_sum,
                  reweight_branches, plot_thresholds, plot_clip_ranges, strict):
    """Stream files and accumulate weighted histograms for all branches in branch_edges.

    Returns {branch: (h, h2)} where h = sum-of-weights and h2 = sum-of-weights-squared.
    For data pass target_total=1.0, raw_w_sum=1.0, reweight_branches=[].
    """
    load_cols = sorted(set(branch_edges.keys())
                       | set(plot_thresholds.keys())
                       | set(plot_clip_ranges.keys())
                       | set(reweight_branches))

    hists = {b: [np.zeros(len(edges) - 1, dtype=float),
                 np.zeros(len(edges) - 1, dtype=float)]
             for b, edges in branch_edges.items()}

    for chunk in _iter_chunks(files, tree_name, load_cols, strict=strict):
        # Compute per-event weights before any filtering.
        raw_w = np.ones(len(chunk), dtype=float)
        for rb in reweight_branches:
            if rb in chunk.columns:
                raw_w *= chunk[rb].to_numpy(dtype=float, copy=False)
        weight = raw_w * (target_total / raw_w_sum) if raw_w_sum > 0 else np.zeros(len(chunk))

        # Drop reweight columns — they are not plot variables.
        drop = [rb for rb in reweight_branches if rb in chunk.columns]
        if drop:
            chunk = chunk.drop(columns=drop)

        # Apply threshold cuts and clip ranges.
        if plot_thresholds:
            mask = _threshold_mask(chunk, plot_thresholds).to_numpy(dtype=bool)
            chunk  = chunk[mask].reset_index(drop=True)
            weight = weight[mask]

        if len(chunk) == 0:
            continue

        if plot_clip_ranges:
            _apply_clip(chunk, plot_clip_ranges)

        # Accumulate histograms.
        for b, edges in branch_edges.items():
            if b not in chunk.columns:
                continue
            h, h2 = _weighted_hist(chunk[b].to_numpy(dtype=float, copy=False), weight, edges)
            hists[b][0] += h
            hists[b][1] += h2

    return {b: (hists[b][0], hists[b][1]) for b in hists}


# -------------------- Threshold and clip filtering --------------------
def _mask_from_cond(col, cond):
    idx = col.index
    if cond is None:
        return pd.Series(True, index=idx)
    if isinstance(cond, (int, float, np.integer, np.floating)):
        return col > float(cond)
    if isinstance(cond, (list, tuple)) and len(cond) == 2 and not isinstance(cond[0], (list, dict, tuple)):
        mn, mx = cond
        m = pd.Series(True, index=idx)
        if mn is not None:
            m &= col > mn
        if mx is not None:
            m &= col < mx
        return m
    if isinstance(cond, (list, tuple)):
        masks = [_mask_from_cond(col, item) for item in cond]
        out = pd.Series(False, index=idx)
        for mask in masks:
            out |= mask
        return out
    if isinstance(cond, dict):
        for key, is_and in (("&", True), ("and", True), ("|", False), ("or", False)):
            if key not in cond:
                continue
            items = cond[key]
            out = pd.Series(True if is_and else False, index=idx)
            for item in items:
                mask = _mask_from_cond(col, item)
                out = (out & mask) if is_and else (out | mask)
            return out
        raise ValueError(f"Unsupported dict condition keys: {cond}")
    raise TypeError(f"Unsupported threshold condition: {cond!r}")


def _threshold_mask(df, thresholds):
    if not thresholds or df is None or len(df) == 0:
        return pd.Series(True, index=df.index if df is not None else None)
    mask = pd.Series(True, index=df.index)
    for b, cond in thresholds.items():
        if b not in df.columns:
            continue
        col = df[b]
        sentinel = col < -990
        mask &= ~sentinel
        mask &= _mask_from_cond(col, cond)
    return mask


def _apply_thresholds(df, thresholds):
    if not thresholds or df is None or len(df) == 0:
        return df
    mask = _threshold_mask(df, thresholds)
    return df.loc[mask].reset_index(drop=True)


def _apply_clip(df, clip_ranges):
    if not clip_ranges or df is None or len(df) == 0:
        return df
    for col, rng in clip_ranges.items():
        if col not in df.columns:
            continue
        arr   = df[col].values.astype(float, copy=True)
        valid = arr >= -990
        lo, hi = rng
        if lo is not None:
            arr[valid & (arr < lo)] = lo
        if hi is not None:
            arr[valid & (arr > hi)] = hi
        df[col] = arr
    return df


def _standardize_model_X(X, clip_ranges, log_transform):
    log_set = set(log_transform)
    for col in X.columns:
        arr = X[col].values.copy()
        sentinel = arr < -990
        valid = ~sentinel
        if not valid.any():
            continue
        lo, hi = clip_ranges.get(col, (None, None))
        if lo is not None:
            arr[valid & (arr < lo)] = lo
        if hi is not None:
            arr[valid & (arr > hi)] = hi
        if col in log_set:
            pos = valid & (arr > 0)
            if pos.any():
                if not np.issubdtype(arr.dtype, np.floating):
                    arr = arr.astype(float)
                arr[pos] = np.log(arr[pos])
        X[col] = arr
    return X


def _drop_decorrelated_features(X, decorrelate):
    if not decorrelate:
        return X
    drop_cols = [name for name in decorrelate if name in X.columns]
    if drop_cols:
        return X.drop(columns=drop_cols)
    return X


def _predict_model_proba(model, X, num_classes):
    return _shared_predict_model_proba(model, X, num_classes)


def _load_score_model(bdt_root, bdt_cfg, tree_name):
    model_pattern = bdt_cfg.get("model_pattern", "{output_root}/{tree_name}_model")
    model_base = model_pattern.format(output_root=bdt_root, tree_name=tree_name)
    class_groups = bdt_cfg["class_groups"]
    return _shared_load_model(
        model_base,
        bdt_cfg,
        len(class_groups),
        log_message=log_message,
    )


def _compare_score_reference(path, feature_names, sample_labels, class_idx, weights, proba):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Prediction reference not found: {path}. Re-run train.py before data_mc.py."
        )

    ref = np.load(path, allow_pickle=False)
    ref_features = ref["feature_names"].astype(str).tolist()
    cur_features = list(feature_names)
    if cur_features != ref_features:
        raise RuntimeError(
            "Prediction reference mismatch for score model features: "
            f"current={cur_features}, reference={ref_features}"
        )

    ref_samples = ref["sample_name"].astype(str)
    cur_samples = np.asarray(sample_labels, dtype=str)
    if not np.array_equal(cur_samples, ref_samples):
        raise RuntimeError("Prediction reference mismatch for score sample order/content")

    ref_class_idx = ref["class_idx"].astype(int)
    cur_class_idx = np.asarray(class_idx, dtype=int)
    if not np.array_equal(cur_class_idx, ref_class_idx):
        raise RuntimeError("Prediction reference mismatch for score class labels")

    ref_weights = ref["weight"].astype(float) * LUMI_TOTAL
    cur_weights = np.asarray(weights, dtype=float)
    weight_rtol = float(ref["weight_rtol"])
    weight_atol = float(ref["weight_atol"])
    if not np.allclose(cur_weights, ref_weights, rtol=weight_rtol, atol=weight_atol):
        diff = float(np.max(np.abs(cur_weights - ref_weights)))
        raise RuntimeError(
            "Prediction reference mismatch for score weights: "
            f"max_abs_diff={diff:.6g}, rtol={weight_rtol}, atol={weight_atol}"
        )

    ref_proba = ref["proba"].astype(float)
    cur_proba = np.asarray(proba, dtype=float)
    proba_rtol = float(ref["proba_rtol"])
    proba_atol = float(ref["proba_atol"])
    if cur_proba.shape != ref_proba.shape:
        raise RuntimeError(
            "Prediction reference mismatch for score probabilities shape: "
            f"current={cur_proba.shape}, reference={ref_proba.shape}"
        )
    if not np.allclose(cur_proba, ref_proba, rtol=proba_rtol, atol=proba_atol):
        diff = float(np.max(np.abs(cur_proba - ref_proba)))
        raise RuntimeError(
            "Prediction reference mismatch for score probabilities: "
            f"max_abs_diff={diff:.6g}, rtol={proba_rtol}, atol={proba_atol}"
        )
    log_message(f"Validated score prediction reference: {path}")


# -------------------- Weight assignment --------------------
def _assign_mc_weight(df, sample_name, tree_entries_total, n_loaded, reweight_branches=None):
    """Assign per-event weight for an MC sample.

    Per event:
        raw_w  = product of reweight_branches (1.0 if empty)
        target_total = lumi_total * xsection * tree_entries_total / raw_entries
        weight = raw_w * target_total / sum(raw_w_loaded)

    So the sample's total weight sums to ``target_total`` regardless of raw_w's
    magnitude; raw_w only shapes the per-event distribution inside the sample.

    Reweight branches are read on raw values (before clip/log/threshold) and
    dropped from ``df`` once raw_w is computed. Computed before any filtering;
    the weights are unchanged afterwards.
    """
    reweight_branches = list(reweight_branches or [])
    if reweight_branches:
        missing = [rb for rb in reweight_branches if rb not in df.columns]
        if missing:
            raise KeyError(
                f"Sample '{sample_name}' missing reweight branches: {', '.join(missing)}"
            )
        raw_w = np.ones(n_loaded, dtype=float)
        for rb in reweight_branches:
            raw_w *= df[rb].to_numpy(dtype=float, copy=False)
        df = df.drop(columns=reweight_branches)
    else:
        raw_w = np.ones(n_loaded, dtype=float)

    info        = SAMPLE_INFO[sample_name]
    xsec        = float(info.get("xsection", 0.0))
    raw_entries = float(info.get("raw_entries", 0.0))
    if raw_entries <= 0.0:
        raise RuntimeError(f"Sample '{sample_name}' has raw_entries={raw_entries}; fill src/sample.json")
    if n_loaded == 0 or tree_entries_total == 0:
        df["weight"] = 0.0
        return df
    target_total = LUMI_TOTAL * xsec * float(tree_entries_total) / raw_entries
    raw_w_sum = float(raw_w.sum())
    if raw_w_sum <= 0.0:
        raise RuntimeError(
            f"Sample '{sample_name}' has non-positive raw weight sum {raw_w_sum:.6g}"
        )
    df["weight"] = raw_w * (target_total / raw_w_sum)
    return df


def _load_test_segments(tree_name, branches, sample_meta):
    parts = []
    for seg in sample_meta["test_segments"]:
        fpath = seg["file"]
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Test split file not found: {fpath}")
        with uproot.open(fpath) as uf:
            if tree_name not in uf:
                raise KeyError(f"Tree '{tree_name}' not in {fpath}")
            tree = uf[tree_name]
            avail = set(tree.keys())
            missing = [branch for branch in branches if branch not in avail]
            if missing:
                raise KeyError(
                    f"Missing branches in {fpath}:{tree_name}: "
                    f"{', '.join(missing[:10])}" + (" ..." if len(missing) > 10 else "")
                )
            parts.append(
                tree.arrays(
                    branches,
                    library="pd",
                    entry_start=int(seg["entry_start"]),
                    entry_stop=int(seg["entry_stop"]),
                )
            )
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


def _assign_test_split_mc_weight(df, sample_name, total_entries, reweight_branches=None):
    reweight_branches = list(reweight_branches or [])
    n_loaded = len(df)
    if reweight_branches:
        missing = [rb for rb in reweight_branches if rb not in df.columns]
        if missing:
            raise KeyError(
                f"Sample '{sample_name}' missing score reweight branches: {', '.join(missing)}"
            )
        raw_w = np.ones(n_loaded, dtype=float)
        for rb in reweight_branches:
            raw_w *= df[rb].to_numpy(dtype=float, copy=False)
        df = df.drop(columns=reweight_branches)
    else:
        raw_w = np.ones(n_loaded, dtype=float)

    info = SAMPLE_INFO[sample_name]
    xsec = float(info.get("xsection", 0.0))
    raw_entries = float(info.get("raw_entries", 0.0))
    if raw_entries <= 0.0:
        raise RuntimeError(f"Sample '{sample_name}' has raw_entries={raw_entries}; fill src/sample.json")
    if n_loaded == 0 or total_entries == 0 or xsec <= 0.0:
        df["weight"] = 0.0
        return df
    raw_w_sum = float(raw_w.sum())
    if raw_w_sum <= 0.0:
        raise RuntimeError(
            f"Sample '{sample_name}' has non-positive score raw weight sum {raw_w_sum:.6g}"
        )
    target_total = LUMI_TOTAL * xsec * float(total_entries) / raw_entries
    df["weight"] = raw_w * (target_total / raw_w_sum)
    return df


def _add_score_columns(df, proba, class_names):
    out = pd.DataFrame({"weight": df["weight"].to_numpy(dtype=float, copy=False)})
    for idx, class_name in enumerate(class_names):
        out[_score_branch_name(class_name)] = proba[:, idx]
    return out


# -------------------- Binning --------------------
def _branch_override(tree_name, branch):
    tree_ov = _tree_plot_cfg(tree_name)
    branches = tree_ov.get("branches", {})
    if isinstance(branches, dict) and branch in branches:
        override = branches.get(branch, {})
        return override if isinstance(override, dict) else {}
    override = tree_ov.get(branch, {})
    return override if isinstance(override, dict) else {}


def _auto_range(arrs, logx):
    mins, maxs = [], []
    for arr in arrs:
        if arr is None:
            continue
        a = np.asarray(arr, dtype=float)
        valid = a[a >= -990]
        if logx:
            valid = valid[valid > 0]
        if valid.size == 0:
            continue
        mins.append(float(valid.min()))
        maxs.append(float(valid.max()))
    if not mins:
        return None
    lo, hi = min(mins), max(maxs)
    if lo >= hi:
        hi = lo + 1.0
    return lo, hi


def _resolve_binning(tree_name, branch, arrs, log_tf_set):
    override = _branch_override(tree_name, branch)
    bins     = int(override.get("bins", DEFAULT_BINS))
    logx     = bool(override.get("logx", False if branch.startswith("score_") else branch in log_tf_set))
    logy     = bool(override.get("logy", True))
    y_range  = tuple(override["y_range"]) if "y_range" in override else None

    if "x_range" in override:
        x_lo, x_hi = override["x_range"]
        x_range = (float(x_lo), float(x_hi))
    elif branch.startswith("score_"):
        x_range = (0.0, 1.0)
    elif isinstance(arrs, tuple):
        # Pre-computed (lo, hi) from the streaming pre-pass.
        lo, hi = arrs
        x_range = (lo, lo + 1.0) if lo >= hi else (lo, hi)
    else:
        x_range = _auto_range(arrs, logx)
        if x_range is None:
            return None
    return bins, x_range, logx, logy, y_range


def _bin_edges(bins, x_range, logx):
    lo, hi = x_range
    if logx:
        if lo <= 0:
            lo = 1e-9
        return np.logspace(math.log10(lo), math.log10(hi), bins + 1)
    return np.linspace(lo, hi, bins + 1)


def _weighted_hist(vals, weights, edges):
    v = np.asarray(vals,    dtype=float)
    w = np.asarray(weights, dtype=float)
    valid = v >= -990
    v = v[valid]
    w = w[valid]
    h,  _ = np.histogram(v, bins=edges, weights=w)
    h2, _ = np.histogram(v, bins=edges, weights=w * w)
    return h.astype(float), h2.astype(float)


# -------------------- Ratio --------------------
def _ratio_data_over_mc(data_vals, data_vars, mc_vals, mc_vars):
    with np.errstate(divide="ignore", invalid="ignore"):
        r  = np.where(mc_vals > 0, data_vals / mc_vals, np.nan)
        tm = np.where(mc_vals   > 0, mc_vars   / np.maximum(mc_vals,   1e-300) ** 2, 0.0)
        td = np.where(data_vals > 0, data_vars / np.maximum(data_vals, 1e-300) ** 2, 0.0)
        sigma = np.abs(r) * np.sqrt(tm + td)
    return r, sigma


# -------------------- Per-tree processing --------------------
def _process_tree(tree_name, no_selection=False):
    log_message(f"Running data_mc.py: tree={tree_name}, no_selection={no_selection}")

    log_message("Loading trained-model config copies")
    bdt_cfg, bdt_br, bdt_sel, test_meta = _bdt_configs_for_tree(tree_name, load_test_meta=not no_selection)
    class_groups     = bdt_cfg["class_groups"]
    class_names      = list(class_groups.keys())
    model_branches   = [item["name"] for item in bdt_br[tree_name]]
    # Score branches require a trained model; skip them in --no-selection mode.
    score_branches   = [] if no_selection else [_score_branch_name(class_name) for class_name in class_names]

    bdt_root_dir   = _bdt_root_for_tree(tree_name)
    bdt_script_dir = os.path.dirname(bdt_root_dir)
    if no_selection:
        # Read directly from the convert output (no mixing step required).
        _conv_dir     = os.path.dirname(_CONVERT_CFG_PATH)
        input_root    = _resolve(convert_cfg["output_root"], _conv_dir)
        # convert uses {output_root} as placeholder; _input_files expects {input_root}
        input_pattern = convert_cfg["output_pattern"].replace("{output_root}", "{input_root}")
    else:
        input_root    = _resolve(bdt_cfg["input_root"], bdt_script_dir)
        input_pattern = bdt_cfg["input_pattern"]

    sel              = bdt_sel.get(tree_name, {})
    clip_ranges      = {k: tuple(v) for k, v in sel.get("clip_ranges", {}).items()}
    thresholds       = {k: (tuple(v) if isinstance(v, list) else v)
                        for k, v in sel.get("thresholds", {}).items()}
    log_tf_set       = set(sel.get("log_transform", []))
    # When --no-selection is active, skip event-level cuts and plot-variable clipping.
    # clip_ranges is still passed to _standardize_model_X so BDT scores stay valid.
    plot_thresholds  = {} if no_selection else thresholds
    plot_clip_ranges = {} if no_selection else clip_ranges

    skip_score = _skip_branches_for_tree(tree_name)
    branches_to_plot = _plot_branches_for_tree(tree_name)
    score_branches = [branch for branch in score_branches if branch not in skip_score]
    for branch in score_branches:
        if branch not in branches_to_plot:
            branches_to_plot.append(branch)
    root_plot_branches = [branch for branch in branches_to_plot if branch not in score_branches]
    reweight_cfg      = plot_cfg.get("event_reweight_branches", {})
    reweight_branches = list(reweight_cfg.get(tree_name, []))
    score_reweight_branches = list(bdt_cfg.get(tree_name, {}).get("event_reweight_branches", []))
    log_message(
        f"Resolved plotting config: branches={len(branches_to_plot)}, "
        f"threshold_branches={len(thresholds)}, clip_branches={len(clip_ranges)}, "
        f"reweight_branches={len(reweight_branches)}, score_branches={len(score_branches)}"
    )

    out_patt = OUTPUT_ROOT_NOSEL_PATT if no_selection else OUTPUT_ROOT_PATT
    out_dir = _resolve(out_patt.format(tree_name=tree_name), _SCRIPT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    log_message(f"Output directory: {out_dir}")

    # Determine which root branches need auto-ranging (no explicit x_range override).
    branch_logx = {b: bool(_branch_override(tree_name, b).get("logx", b in log_tf_set))
                   for b in root_plot_branches}
    auto_range_branches = [b for b in root_plot_branches
                           if "x_range" not in _branch_override(tree_name, b)
                           and not b.startswith("score_")]

    # Pre-pass over MC samples: collect value ranges and per-sample raw-weight sums.
    # These replace loading all events into memory; the ranges feed binning resolution
    # and raw_w_sums feed per-event weight normalisation during streaming.
    n_mc_samples = sum(len(s) for s in class_groups.values())
    log_message(f"Pre-pass: {n_mc_samples} MC samples, {len(DATA_SAMPLES)} data samples")
    mc_raw_w_sums  = {}
    mc_n_totals    = {}
    mc_sample_files = {}
    prepass_mins   = {b:  np.inf for b in auto_range_branches}
    prepass_maxs   = {b: -np.inf for b in auto_range_branches}

    for cls_name, samples in class_groups.items():
        for sname in samples:
            if sname not in SAMPLE_INFO:
                raise RuntimeError(f"MC sample '{sname}' not found in sample.json")
            files = _input_files(sname, input_root, input_pattern)
            if not files:
                raise RuntimeError(f"No ROOT files found for MC sample '{sname}'")
            n_total = _tree_entries_total(files, tree_name)
            if n_total <= 0:
                raise RuntimeError(f"Empty tree '{tree_name}' for MC sample '{sname}'")
            mc_n_totals[sname]     = n_total
            mc_sample_files[sname] = files
            rw_sum, ranges = _prepass(
                files, tree_name, auto_range_branches, reweight_branches,
                branch_logx, strict=not no_selection,
            )
            # If there are no reweight branches raw_w=1 per event so rw_sum = n_loaded;
            # fall back to n_total in case the pre-pass found nothing.
            mc_raw_w_sums[sname] = rw_sum if rw_sum > 0 else float(n_total)
            for b, (lo, hi) in ranges.items():
                prepass_mins[b] = min(prepass_mins[b], lo)
                prepass_maxs[b] = max(prepass_maxs[b], hi)
            log_message(f"  {sname}: n_total={n_total}, raw_w_sum={mc_raw_w_sums[sname]:.6g}")

    # Pre-pass over data (weight=1 everywhere, so only range collection matters).
    data_sample_files = {}
    for sname in DATA_SAMPLES:
        files = _input_files(sname, input_root, input_pattern)
        if not files:
            raise RuntimeError(f"No ROOT files found for data sample '{sname}'")
        data_sample_files[sname] = files
        _, ranges = _prepass(
            files, tree_name, auto_range_branches, [],
            branch_logx, strict=not no_selection,
        )
        for b, (lo, hi) in ranges.items():
            prepass_mins[b] = min(prepass_mins[b], lo)
            prepass_maxs[b] = max(prepass_maxs[b], hi)
        log_message(f"  data {sname}: pre-pass done")

    merged_ranges = {}
    for b in auto_range_branches:
        lo, hi = prepass_mins[b], prepass_maxs[b]
        if np.isfinite(lo) and np.isfinite(hi) and lo <= hi:
            merged_ranges[b] = (lo, hi)

    # Build derived model score branches. MC scores use the saved test split and
    # are validated against train.py's signal-region reference; data scores use
    # the full configured data input, matching the ordinary branch plots.
    score_class_dfs = {}
    score_data_df = None
    if score_branches:
        log_message("Preparing model score branches")
        clf = _load_score_model(bdt_root_dir, bdt_cfg, tree_name)
        decorrelate = list(bdt_cfg.get(tree_name, {}).get("decorrelate", []))
        score_load = sorted(set(model_branches) | set(thresholds.keys()) | set(score_reweight_branches))
        sample_to_class_name = {}
        sample_to_class_idx = {}
        for idx, (cls_name, samples) in enumerate(class_groups.items()):
            for sample_name in samples:
                sample_to_class_name[sample_name] = cls_name
                sample_to_class_idx[sample_name] = idx

        score_parts_by_class = {cls_name: [] for cls_name in class_names}
        ref_sample_labels = []
        ref_class_idx = []
        ref_weights = []
        ref_proba_parts = []
        ref_feature_names = None

        log_message(f"Loading MC score test split samples: n={len(test_meta['samples'])}")
        for sample_name, sample_meta in test_meta["samples"].items():
            if sample_name not in sample_to_class_name:
                raise RuntimeError(f"Test split sample '{sample_name}' is not in class_groups")
            df = _load_test_segments(tree_name, score_load, sample_meta)
            if df is None or len(df) == 0:
                raise RuntimeError(f"No test split events loaded for sample '{sample_name}'")
            df = _assign_test_split_mc_weight(
                df,
                sample_name,
                int(sample_meta["total_entries"]),
                score_reweight_branches,
            )
            mask = _threshold_mask(df, plot_thresholds)
            df = df.loc[mask].reset_index(drop=True)
            if len(df) == 0:
                log_message(f"  [WARN] score sample '{sample_name}' has zero events after filtering")
                continue
            X_model = _standardize_model_X(df[model_branches].copy(), clip_ranges, list(log_tf_set))
            X_model = _drop_decorrelated_features(X_model, decorrelate)
            proba = _predict_model_proba(clf, X_model, len(class_names))
            if ref_feature_names is None:
                ref_feature_names = list(X_model.columns)
            score_df = _add_score_columns(df, proba, class_names)
            cls_name = sample_to_class_name[sample_name]
            score_parts_by_class[cls_name].append(score_df)
            ref_sample_labels.extend([sample_name] * len(df))
            ref_class_idx.extend([sample_to_class_idx[sample_name]] * len(df))
            ref_weights.extend(score_df["weight"].to_numpy(dtype=float, copy=False))
            ref_proba_parts.append(proba)
            log_message(
                f"  score {sample_name}: class={cls_name}, test_loaded={len(df)}, "
                f"weight_sum={float(score_df['weight'].sum()):.6g}"
            )

        for cls_name, parts in score_parts_by_class.items():
            if parts:
                score_class_dfs[cls_name] = pd.concat(parts, ignore_index=True)

        if not ref_proba_parts:
            raise RuntimeError(f"No MC score events after filtering for tree '{tree_name}'")
        score_proba_ref = np.concatenate(ref_proba_parts, axis=0)
        _compare_score_reference(
            os.path.join(bdt_root_dir, "test_reference_signal_region.npz"),
            ref_feature_names,
            ref_sample_labels,
            ref_class_idx,
            ref_weights,
            score_proba_ref,
        )

        if DATA_SAMPLES:
            score_data_load = sorted(set(model_branches) | set(thresholds.keys()))
            score_data_parts = []
            log_message(f"Loading data score samples: n={len(DATA_SAMPLES)}")
            for sname in DATA_SAMPLES:
                files = _input_files(sname, input_root, input_pattern)
                if not files:
                    raise RuntimeError(f"No ROOT files found for data score sample '{sname}'")
                df = _load_tree(files, tree_name, score_data_load)
                if df is None or len(df) == 0:
                    log_message(f"  [WARN] data score sample '{sname}' has zero entries")
                    continue
                df = _apply_thresholds(df, plot_thresholds)
                if df is None or len(df) == 0:
                    log_message(f"  [WARN] data score sample '{sname}' has zero events after filtering")
                    continue
                df["weight"] = 1.0
                X_model = _standardize_model_X(df[model_branches].copy(), clip_ranges, list(log_tf_set))
                X_model = _drop_decorrelated_features(X_model, decorrelate)
                proba = _predict_model_proba(clf, X_model, len(class_names))
                score_data_parts.append(_add_score_columns(df, proba, class_names))
                log_message(f"  data score {sname}: events={len(df)}")
            if score_data_parts:
                score_data_df = pd.concat(score_data_parts, ignore_index=True)
                log_message(f"Loaded data score events: {len(score_data_df)}")
            else:
                log_message("Loaded data score events: 0")

    # Resolve binning for all root branches using pre-pass ranges.
    log_message("Resolving binning for all branches")
    branch_binning = {}
    for b in root_plot_branches:
        precomp = merged_ranges.get(b)   # None for explicit-range branches (override takes priority)
        binning = _resolve_binning(tree_name, b, precomp, log_tf_set)
        if binning is None:
            log_message(f"  [WARN] no range for {tree_name}:{b}, will skip")
        else:
            branch_binning[b] = binning
    branch_edges = {b: _bin_edges(bins, x_range, logx)
                    for b, (bins, x_range, logx, logy, y_range) in branch_binning.items()}
    log_message(f"  {len(branch_binning)}/{len(root_plot_branches)} root branches have valid binning")

    # Streaming histogram accumulation for MC.
    log_message(f"Streaming MC histograms ({len(class_groups)} classes)")
    mc_hists = {cls: {} for cls in class_names}
    for cls_name, samples in class_groups.items():
        log_message(f"  Class '{cls_name}' ({len(samples)} samples)")
        for sname in samples:
            info         = SAMPLE_INFO[sname]
            xsec         = float(info.get("xsection", 0.0))
            raw_entries  = float(info.get("raw_entries", 0.0))
            n_total      = mc_n_totals[sname]
            raw_w_sum    = mc_raw_w_sums[sname]
            target_total = (LUMI_TOTAL * xsec * float(n_total) / raw_entries
                            if raw_entries > 0 else 0.0)
            sample_hists = _stream_hists(
                mc_sample_files[sname], tree_name, branch_edges,
                target_total, raw_w_sum, reweight_branches,
                plot_thresholds, plot_clip_ranges,
                strict=not no_selection,
            )
            for b, (h, h2) in sample_hists.items():
                if b not in mc_hists[cls_name]:
                    mc_hists[cls_name][b] = [h.copy(), h2.copy()]
                else:
                    mc_hists[cls_name][b][0] += h
                    mc_hists[cls_name][b][1] += h2
            log_message(f"    {sname}: target_total={target_total:.6g}")
        mc_hists[cls_name] = {b: (v[0], v[1]) for b, v in mc_hists[cls_name].items()}

    # Streaming histogram accumulation for data.
    log_message(f"Streaming data histograms ({len(DATA_SAMPLES)} samples)")
    data_hists = {}
    for sname in DATA_SAMPLES:
        sample_hists = _stream_hists(
            data_sample_files[sname], tree_name, branch_edges,
            target_total=1.0, raw_w_sum=1.0, reweight_branches=[],
            plot_thresholds=plot_thresholds, plot_clip_ranges=plot_clip_ranges,
            strict=not no_selection,
        )
        for b, (h, h2) in sample_hists.items():
            if b not in data_hists:
                data_hists[b] = [h.copy(), h2.copy()]
            else:
                data_hists[b][0] += h
                data_hists[b][1] += h2
        log_message(f"  data {sname}: streaming done")
    data_hists = {b: (v[0], v[1]) for b, v in data_hists.items()}

    # Plot each requested branch.
    log_message(f"Plotting branches: total={len(branches_to_plot)}")
    palette = plt.rcParams["axes.prop_cycle"].by_key().get(
        "color", ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    )
    color_map = {c: palette[i % len(palette)] for i, c in enumerate(class_names)}

    for plot_idx, branch in enumerate(branches_to_plot, start=1):
        log_message(f"Plotting branch {plot_idx}/{len(branches_to_plot)}: {branch}")
        is_score_branch = branch in score_branches

        if is_score_branch:
            # Score branches: binning and histograms computed from test-split DataFrames.
            plot_class_dfs = score_class_dfs
            plot_data_df   = score_data_df
            arrs = []
            for cls in class_names:
                if cls in plot_class_dfs and branch in plot_class_dfs[cls].columns:
                    arrs.append(plot_class_dfs[cls][branch].values)
            if plot_data_df is not None and branch in plot_data_df.columns:
                arrs.append(plot_data_df[branch].values)
            binning = _resolve_binning(tree_name, branch, arrs, log_tf_set)
            if binning is None:
                log_message(f"  [WARN] no data for {tree_name}:{branch}, skipping")
                continue
            bins, x_range, logx, logy, y_range = binning
            edges       = _bin_edges(bins, x_range, logx)
            bin_centers = 0.5 * (edges[:-1] + edges[1:])
            bin_widths  = edges[1:] - edges[:-1]
            mc_total_v  = np.zeros(bins)
            mc_total_w2 = np.zeros(bins)
            mc_per_cls  = {}
            mc_yields   = {}
            for cls in class_names:
                if cls in plot_class_dfs and branch in plot_class_dfs[cls].columns:
                    h, h2 = _weighted_hist(
                        plot_class_dfs[cls][branch].values,
                        plot_class_dfs[cls]["weight"].values, edges,
                    )
                else:
                    h, h2 = np.zeros(bins), np.zeros(bins)
                mc_per_cls[cls] = (h, h2)
                mc_total_v  += h
                mc_total_w2 += h2
                mc_yields[cls] = float(h.sum())
            if plot_data_df is not None and branch in plot_data_df.columns:
                data_v, data_w2 = _weighted_hist(
                    plot_data_df[branch].values, plot_data_df["weight"].values, edges,
                )
            else:
                data_v, data_w2 = np.zeros(bins), np.zeros(bins)
        else:
            # Root branches: look up pre-computed streaming histograms.
            if branch not in branch_binning:
                log_message(f"  [WARN] no range for {tree_name}:{branch}, skipping")
                continue
            bins, x_range, logx, logy, y_range = branch_binning[branch]
            edges       = branch_edges[branch]
            bin_centers = 0.5 * (edges[:-1] + edges[1:])
            bin_widths  = edges[1:] - edges[:-1]
            mc_total_v  = np.zeros(bins)
            mc_total_w2 = np.zeros(bins)
            mc_per_cls  = {}
            mc_yields   = {}
            for cls in class_names:
                h, h2 = mc_hists[cls].get(branch, (np.zeros(bins), np.zeros(bins)))
                mc_per_cls[cls] = (h, h2)
                mc_total_v  += h
                mc_total_w2 += h2
                mc_yields[cls] = float(h.sum())
            data_v, data_w2 = data_hists.get(branch, (np.zeros(bins), np.zeros(bins)))

        fig, (ax, axr) = plt.subplots(
            2, 1, figsize=(10, 10),
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0},
            sharex=True,
        )

        # Draw the stacked MC histograms from low to high total yield.
        order = np.argsort([mc_yields[c] for c in class_names])
        bottom = np.zeros(bins)
        for idx in order:
            cls = class_names[idx]
            h, _ = mc_per_cls[cls]
            ax.bar(
                edges[:-1], h, width=bin_widths, bottom=bottom,
                align="edge", color=color_map[cls], edgecolor="none",
                linewidth=0, antialiased=False, alpha=0.9, label=cls,
            )
            bottom += h
        ax.margins(x=0)

        # Draw the total MC uncertainty band.
        mc_sigma = np.sqrt(np.maximum(mc_total_w2, 0.0))
        lower = np.clip(mc_total_v - mc_sigma, 1e-12, None)
        upper = np.clip(mc_total_v + mc_sigma, 1e-12, None)
        ax.fill_between(
            bin_centers, lower, upper, step="mid",
            facecolor="none", edgecolor="gray", hatch="///", linewidth=0,
        )

        # Draw the data points.
        data_sigma = np.sqrt(np.maximum(data_w2, 0.0))
        y_plot = np.where(data_v > 0, data_v, np.nan)
        ax.errorbar(
            bin_centers, y_plot, yerr=data_sigma,
            fmt="o", ms=7.6, color="black", mfc="black", mec="black",
            elinewidth=1.5, capsize=0, label="Data",
        )

        # Configure the axes.
        if logx:
            ax.set_xscale("log")
            axr.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlim(*x_range)
        axr.set_xlim(*x_range)

        if y_range is not None:
            ax.set_ylim(*y_range)
        else:
            vis = (mc_total_v > 0) | (data_v > 0)
            if np.any(vis):
                ymax = max(float(np.max(mc_total_v[vis])), float(np.max(data_v[vis])))
            else:
                ymax = 1.0
            if logy:
                ax.set_ylim(0.1, max(1.0, ymax * 5.0))
            else:
                ax.set_ylim(0.0, max(1.0, ymax * 1.3))

        ax.set_ylabel("Events", fontsize=24)
        hep.cms.label("Preliminary", data=True, com=13.6, year="2024", lumi=LUMI_TOTAL, ax=ax)

        handles, labels = ax.get_legend_handles_labels()
        if "Data" in labels:
            i = labels.index("Data")
            handles.append(handles.pop(i))
            labels.append(labels.pop(i))
        ax.legend(handles, labels, loc="best", fontsize=17, frameon=False, ncol=2)

        # Draw the Data/MC ratio panel.
        ratio, r_err = _ratio_data_over_mc(data_v, data_w2, mc_total_v, mc_total_w2)
        axr.errorbar(
            bin_centers, ratio, yerr=r_err,
            fmt="o", ms=7.6, color="black", mfc="black", mec="black",
            elinewidth=1.5, capsize=0,
        )
        axr.axhline(1.0, color="black", linestyle="--", linewidth=1.5)

        finite = np.isfinite(ratio)
        if np.any(finite):
            safe_err = np.nan_to_num(r_err[finite], nan=0.0)
            rmax = float(np.nanmax(ratio[finite] + safe_err))
            rmin = float(np.nanmin(ratio[finite] - safe_err))
            if not np.isfinite(rmax) or rmax <= 0:
                rmax = 1.0
            if rmax < 5.0:
                axr.set_ylim(max(0.0, 0.8 * rmin), 1.2 * rmax)
            else:
                axr.set_ylim(0.0, 5.0)
        else:
            axr.set_ylim(0.0, 2.0)

        axr.set_ylabel(r"$\frac{Data}{MC}$", fontsize=26)
        axr.yaxis.set_label_coords(-0.05, 0.6)
        axr.set_xlabel(branch, fontsize=24)

        out_path = os.path.join(out_dir, f"{tree_name}_{branch}.pdf")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        log_message(f"Wrote plot file: {out_path}")
    log_message(f"Finished data_mc.py for tree={tree_name}")


def main():
    parser = argparse.ArgumentParser(description="Data vs MC comparison plotter")
    parser.add_argument(
        "--no-selection", action="store_true",
        help="Skip threshold and clip-range cuts on plot variables (writes to output_root_nosel).",
    )
    args = parser.parse_args()

    out_patt = OUTPUT_ROOT_NOSEL_PATT if args.no_selection else OUTPUT_ROOT_PATT
    log_message(
        f"Running data_mc.py: trees={','.join(SUBMIT_TREES)}, "
        f"bdt_root={BDT_ROOT_PATT}, output_root={out_patt}, "
        f"no_selection={args.no_selection}"
    )
    for tree_name in SUBMIT_TREES:
        _process_tree(tree_name, no_selection=args.no_selection)


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        log_message(f"Runtime error: {ex}")
        raise
