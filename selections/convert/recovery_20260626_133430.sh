#!/usr/bin/env bash
set -e
# Recovery for 25 sample(s) generated 20260626_133430
cd /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools

# 2024B: OUTPUT_MISSING — reprocess + merge (temps incomplete/removed)
python3 run.py 0 2024B --slurm

# 2024I: OUTPUT_MISSING — reprocess + merge (temps incomplete/removed)
python3 run.py 0 2024I --slurm

# zzz: STALE_SCHEMA — reprocess + merge (temps incomplete/removed)
python3 run.py 0 zzz --slurm

# qcd_ht600to800: STALE_SCHEMA — all 12 temps valid, just re-merge
CONVERT_CONFIG_PATH=/depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert/config.json /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert/convert_branch qcd_ht600to800 --merge-successful-batches

# qcd_ht800to1000: STALE_SCHEMA — all 12 temps valid, just re-merge
CONVERT_CONFIG_PATH=/depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert/config.json /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert/convert_branch qcd_ht800to1000 --merge-successful-batches

# qcd_ht1000to1200: STALE_SCHEMA — reprocess + merge (temps incomplete/removed)
python3 run.py 0 qcd_ht1000to1200 --slurm

# qcd_ht1200to1500: STALE_SCHEMA — all 11 temps valid, just re-merge
CONVERT_CONFIG_PATH=/depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert/config.json /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert/convert_branch qcd_ht1200to1500 --merge-successful-batches

# qcd_ht1500to2000: STALE_SCHEMA — all 22 temps valid, just re-merge
CONVERT_CONFIG_PATH=/depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert/config.json /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert/convert_branch qcd_ht1500to2000 --merge-successful-batches

# qcd_ht2000: STALE_SCHEMA — all 18 temps valid, just re-merge
CONVERT_CONFIG_PATH=/depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert/config.json /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert/convert_branch qcd_ht2000 --merge-successful-batches

# ttbar_had: STALE_SCHEMA — all 192 temps valid, just re-merge
CONVERT_CONFIG_PATH=/depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert/config.json /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert/convert_branch ttbar_had --merge-successful-batches

# ttbar_semilep: STALE_SCHEMA — all 45 temps valid, just re-merge
CONVERT_CONFIG_PATH=/depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert/config.json /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert/convert_branch ttbar_semilep --merge-successful-batches

# wjets_h100to400: INCOMPLETE — reprocess + merge (temps incomplete/removed)
python3 run.py 0 wjets_h100to400 --slurm

# wjets_h400to800: STALE_SCHEMA — all 15 temps valid, just re-merge
CONVERT_CONFIG_PATH=/depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert/config.json /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert/convert_branch wjets_h400to800 --merge-successful-batches

# wjets_h800to1500: STALE_SCHEMA — reprocess + merge (temps incomplete/removed)
python3 run.py 0 wjets_h800to1500 --slurm

# wjets_h1500to2500: STALE_SCHEMA — all 43 temps valid, just re-merge
CONVERT_CONFIG_PATH=/depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert/config.json /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert/convert_branch wjets_h1500to2500 --merge-successful-batches

# wjets_h2500: STALE_SCHEMA — reprocess + merge (temps incomplete/removed)
python3 run.py 0 wjets_h2500 --slurm

# zjets_h400to800: STALE_SCHEMA — reprocess + merge (temps incomplete/removed)
python3 run.py 0 zjets_h400to800 --slurm

# zjets_h800to1500: STALE_SCHEMA — all 46 temps valid, just re-merge
CONVERT_CONFIG_PATH=/depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert/config.json /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert/convert_branch zjets_h800to1500 --merge-successful-batches

# zjets_h1500to2500: STALE_SCHEMA — all 62 temps valid, just re-merge
CONVERT_CONFIG_PATH=/depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert/config.json /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert/convert_branch zjets_h1500to2500 --merge-successful-batches

# zjets_h2500: STALE_SCHEMA — all 47 temps valid, just re-merge
CONVERT_CONFIG_PATH=/depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert/config.json /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert/convert_branch zjets_h2500 --merge-successful-batches

# dy_h800to1500_m50to120: TOTAL_FAILURE — reprocess + merge (temps incomplete/removed)
python3 run.py 0 dy_h800to1500_m50to120 --slurm

# wlnu_h400to800_m0to120: STALE_SCHEMA — all 8 temps valid, just re-merge
CONVERT_CONFIG_PATH=/depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert/config.json /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert/convert_branch wlnu_h400to800_m0to120 --merge-successful-batches

# WWToLNu2Q: INCOMPLETE — reprocess + merge (temps incomplete/removed)
python3 run.py 0 WWToLNu2Q --slurm

# ZZto2Nu2Q: OUTPUT_MISSING — reprocess + merge (temps incomplete/removed)
python3 run.py 0 ZZto2Nu2Q --slurm

# data_2024: MERGE_CORRUPT — reprocess + merge (temps incomplete/removed)
python3 run.py 0 data_2024 --slurm

