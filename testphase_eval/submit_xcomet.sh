#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH -J xcomet
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err

set -o xtrace
set -e

source ../.venv-comet/bin/activate
uv run --active --no-sync python3 run_xcomet_ukr.py --teams-json "WMT2026_submission_and_gold/wmt2026_teams.json" --submissions-dir "WMT2026_submission_and_gold/WMT2026_submissions/Ukrainian" --gold-dir "WMT2026_submission_and_gold/test_data"
