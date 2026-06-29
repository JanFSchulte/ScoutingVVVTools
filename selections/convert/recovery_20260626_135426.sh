#!/usr/bin/env bash
set -e
# Recovery for 12 sample(s) generated 20260626_135426
cd /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools

# zzz: STALE_SCHEMA — reprocess + merge (temps incomplete/removed)
python3 run.py 0 zzz --slurm

# qcd_ht1000to1200: INCOMPLETE — reprocess + merge (temps incomplete/removed)
python3 run.py 0 qcd_ht1000to1200 --slurm

# wjets_h100to400: INCOMPLETE — reprocess + merge (temps incomplete/removed)
python3 run.py 0 wjets_h100to400 --slurm

# wjets_h800to1500: INCOMPLETE — reprocess + merge (temps incomplete/removed)
python3 run.py 0 wjets_h800to1500 --slurm

# wjets_h2500: INCOMPLETE — reprocess + merge (temps incomplete/removed)
python3 run.py 0 wjets_h2500 --slurm

# zjets_h400to800: INCOMPLETE — reprocess + merge (temps incomplete/removed)
python3 run.py 0 zjets_h400to800 --slurm

# dy_h800to1500_m50to120: TOTAL_FAILURE — reprocess + merge (temps incomplete/removed)
python3 run.py 0 dy_h800to1500_m50to120 --slurm

# WWToLNu2Q: INCOMPLETE — reprocess + merge (temps incomplete/removed)
python3 run.py 0 WWToLNu2Q --slurm

# ZZto2Nu2Q: OUTPUT_MISSING — reprocess + merge (temps incomplete/removed)
python3 run.py 0 ZZto2Nu2Q --slurm

# data_2024: INCOMPLETE — reprocess + merge (temps incomplete/removed)
python3 run.py 0 data_2024 --slurm

