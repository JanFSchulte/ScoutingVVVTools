#!/usr/bin/env python3
"""SR granularity study of one study variant: number of SRs x minimum background MC events.

    python3 sr_granularity.py submit [--variant shallow_cap20]   # write configs, submit one job per point
    python3 sr_granularity.py summary [--variant shallow_cap20]  # table + sr_granularity/<variant>/summary.json

Every grid point (N_SRS x MIN_BKG_ENTRIES) gets sr_granularity/<variant>/n<N>_minb<M>/ and one SLURM
job that runs, in sequence:
  1. signal_region_hist.py on the eval split (the variant's sr_config.json with n_signal_regions and
     min_bkg_entries changed),
  2. qcd_est.py on the test split in abcd_mode per_sr (the variant's qcd_est_per_sr.json with the
     B/D sideband as the complement of the SR union, robust for small SRs),
  3. combine.C for the combined scenario only (scenario_scopes), MC-true QCD and per-SR ABCD, with
     autoMCStats, the zero-count floors, the ABCD non-closure and lumi. The theory/pileup JSONs are left
     out: they belong to older SRs.
The combine binary is sr_granularity/combine_run (built from combine/combine.C).
"""
import argparse, csv, json, math, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
SR_DIR = os.path.join(REPO, "selections", "signal_region")
STUDY = os.path.join(HERE, "sr_granularity")
COMBINE_BIN = os.path.join(STUDY, "combine_run")
N_SRS = [1, 2, 3, 4]
# combine nuisances of the study (combine.C requires the input of every enabled nuisance, and the
# theory/pileup/jet-systematics JSONs belong to older SRs).
STUDY_NUISANCES = ["lumi", "autoMCStats", "zero_count_floor", "qcd_abcd_stat", "qcd_abcd_nonclosure"]
MIN_BKG_ENTRIES = [0, 30, 100, 300, 1000]


def point_dir(variant, n, m):
    return os.path.join(STUDY, variant, f"n{n}_minb{m}")


def write_configs(variant, n, m):
    vdir = os.path.join(HERE, variant)
    d = point_dir(variant, n, m)
    os.makedirs(d, exist_ok=True)
    sr = json.load(open(os.path.join(vdir, "sr_config.json")))
    sr.update({"n_signal_regions": n, "min_bkg_entries": m, "output_dir": os.path.join(d, "sr")})
    json.dump(sr, open(os.path.join(d, "sr_config.json"), "w"), indent=2)
    qe = json.load(open(os.path.join(vdir, "qcd_est_per_sr.json")))
    qe.update({"bdt_root": os.path.join(vdir, TREE), "signal_region_csv": os.path.join(d, "sr", "signal_region.csv"),
               "output_dir": os.path.join(d, "qcd_est"), "abcd_mode": "per_sr", "abcd_sideband_mode": "complement"})
    json.dump(qe, open(os.path.join(d, "qcd_est.json"), "w"), indent=2)
    cb = json.load(open(os.path.join(REPO, "combine", "config.json")))
    for key in ("theory_syst_json", "pileup_syst_json", "jes_syst_json", "jer_syst_json"):
        cb.pop(key, None)
    cb.update({"channels": [{"name": TREE, "root_file": os.path.join(d, "qcd_est", "qcd_abcd_yields.root"),
                             "bdt_root": os.path.join(vdir, TREE)}],
               "output_dir": os.path.join(d, "combine"), "scenario_scopes": ["combined"], "enabled_nuisances": STUDY_NUISANCES})
    json.dump(cb, open(os.path.join(d, "combine.json"), "w"), indent=2)
    return d


def submit(variant, account, exclude):
    if not os.path.exists(COMBINE_BIN):
        sys.exit(f"missing {COMBINE_BIN}: build it from combine/combine.C first")
    for n in N_SRS:
        for m in MIN_BKG_ENTRIES:
            d = write_configs(variant, n, m)
            cmd = (f"cd {SR_DIR} && python3 signal_region_hist.py {d}/sr_config.json"
                   f" && QCD_EST_CONFIG_PATH={d}/qcd_est.json python3 {REPO}/background_estimation/qcd_est.py"
                   f" && cd {d} && COMBINE_CONFIG_PATH={d}/combine.json {COMBINE_BIN}")
            args = ["sbatch", "--parsable", f"--job-name=srgran_n{n}_m{m}", f"--account={account}",
                    "--cpus-per-task=8", "--mem=16G", "--time=06:00:00", f"--output={d}/job.log", f"--wrap={cmd}"]
            if exclude:
                args.insert(-1, f"--exclude={exclude}")
            job = subprocess.check_output(args, text=True).strip()
            print(f"n={n} min_bkg_entries={m}: job {job} -> {d}")


def z_simple(s, b):
    return math.sqrt(2 * ((s + b) * math.log(1 + s / b) - s)) if b > 0 and s > 0 else 0.0


def z_sigma(s, b, sig2):
    """Asimov Z with background uncertainty (as signal_region.asimov_z)."""
    if s <= 0 or b <= 0:
        return 0.0
    if sig2 <= 0:
        return z_simple(s, b)
    t1 = (s + b) * math.log((s + b) * (b + sig2) / (b * b + (s + b) * sig2))
    t2 = (b * b / sig2) * math.log(1 + sig2 * s / (b * (b + sig2)))
    return math.sqrt(max(2 * (t1 - t2), 0.0))


def read_csv_value(path, column):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        for row in csv.DictReader(fh):
            if row["scope"] == "combined":
                return float(row[column])
    return None


def summary(variant):
    import uproot
    rows = []
    for n in N_SRS:
        for m in MIN_BKG_ENTRIES:
            d = point_dir(variant, n, m)
            row = {"n_srs": n, "min_bkg_entries": m}
            sr_csv = os.path.join(d, "sr", "signal_region.csv")
            if os.path.exists(sr_csv):
                with open(sr_csv) as fh:
                    srs = list(csv.DictReader(fh))
                row["eval_z_sigma"] = math.sqrt(sum(z_sigma(float(r["S"]), float(r["B"]), float(r["B_err"]) ** 2) ** 2 for r in srs))
                row["eval_min_bkg_mc"] = min(int(float(r["B_entries"])) for r in srs)
            root = os.path.join(d, "qcd_est", "qcd_abcd_yields.root")
            if os.path.exists(root):
                f = uproot.open(root)
                ids = [int(x) for x in f["metadata/signal_regions"]["bin_index"].array(library="np")]
                groups = sorted({k.split("/")[1].split(";")[0] for k in f.keys() if k.startswith("groups/")})
                sig = [g for g in groups if g in ("vvv", "vh")]
                z2_simple = z2_sigma = 0.0
                min_mc = None
                for i in ids:
                    get = lambda g, h: float(f[f"groups/{g}/sr{i}/{h}"].values()[0])
                    s = sum(get(g, "yield") for g in sig)
                    b = sum(get(g, "yield") for g in groups if g not in sig)
                    s2 = sum(get(g, "stat_error") ** 2 for g in groups if g not in sig)
                    nmc = sum(get(g, "n_entries") for g in groups if g not in sig)
                    z2_simple += z_simple(s, b) ** 2
                    z2_sigma += z_sigma(s, b, s2) ** 2
                    min_mc = nmc if min_mc is None else min(min_mc, nmc)
                row.update(test_z_simple=math.sqrt(z2_simple), test_z_sigma=math.sqrt(z2_sigma),
                           test_min_bkg_mc=int(min_mc), abcd_tf=float(f["qcd_predict/abcd_tf"].values()[0]))
            cdir = os.path.join(d, "combine")
            row["combine_z_mc"] = read_csv_value(os.path.join(cdir, "significance.csv"), "significance")
            row["combine_z_abcd"] = read_csv_value(os.path.join(cdir, "significance_abcd_mc.csv"), "significance")
            row["combine_limit_mc"] = read_csv_value(os.path.join(cdir, "limits.csv"), "exp_50")
            row["combine_limit_abcd"] = read_csv_value(os.path.join(cdir, "limits_abcd_mc.csv"), "exp_50")
            rows.append(row)
    os.makedirs(os.path.join(STUDY, variant), exist_ok=True)
    json.dump(rows, open(os.path.join(STUDY, variant, "summary.json"), "w"), indent=1)
    fmt = lambda v, p=2: "-" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))
    print(f"{'N':>2} {'minMC':>5} | {'eval Z':>6} {'MC':>4} | {'test Z':>6} {'Z(sB)':>6} {'MC':>4} | "
          f"{'comb Z(MC QCD)':>14} {'r<':>6} | {'comb Z(ABCD)':>12} {'r<':>6}")
    for r in rows:
        print(f"{r['n_srs']:>2} {r['min_bkg_entries']:>5} | {fmt(r.get('eval_z_sigma')):>6} {fmt(r.get('eval_min_bkg_mc')):>4} | "
              f"{fmt(r.get('test_z_simple')):>6} {fmt(r.get('test_z_sigma')):>6} {fmt(r.get('test_min_bkg_mc')):>4} | "
              f"{fmt(r.get('combine_z_mc')):>14} {fmt(r.get('combine_limit_mc'), 1):>6} | "
              f"{fmt(r.get('combine_z_abcd')):>12} {fmt(r.get('combine_limit_abcd'), 1):>6}")


TREE = "boosted3"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["submit", "summary"])
    parser.add_argument("--variant", default="shallow_cap20")
    parser.add_argument("--account", default="cms-express")
    parser.add_argument("--exclude", default="hammer-f015")
    a = parser.parse_args()
    submit(a.variant, a.account, a.exclude) if a.action == "submit" else summary(a.variant)
