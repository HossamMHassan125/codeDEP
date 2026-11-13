#!/bin/bash
# Redesigned Monte Carlo study (scenarios of Section 6). NSIM = replications per cell.
# The reported run used NSIM=50 because of compute limits; use NSIM>=1000 on a cluster (cells are independent).
NSIM=${1:-50}
cd "$(dirname "$0")"
python3 run_sim.py S0  200 $NSIM
python3 run_sim.py S1b 200 $NSIM
python3 run_sim.py S2  200 $NSIM
python3 run_sim.py S3  200 $NSIM
python3 run_sim.py S1a 200 $NSIM
python3 run_sim.py S0b 200 $NSIM
