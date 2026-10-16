#!/bin/bash
# Paper D: re-train every network and re-run every learned-vs-exact comparison from the seeds.
# NOT needed for the figures, tables or verification (those use the stored weights in results/o1_nets, results/p1_nets).
# Run from paperD/:   bash retrain/retrain_all.sh
# The training data are regenerated from fixed seeds (results/*_nets/data_*.npy, ~160 MB each, not shipped).
# Cost on one CPU core (OMP_NUM_THREADS=1): 2-d H128 B1024 4.0 ms/step, H256 12.2 ms/step, B4096 19.1 ms/step; 4-d 3.8 ms/step
# (results/q1_bench.json); 300k steps = 0.3-1.6 h per network; B16384 (R1) ~ 4x B4096 per step.
# Outputs are written to results/ under the SAME names as the shipped files; move the shipped results aside first if you want
# to compare old and new.  Each line is independent (one output file per process), so lines may run in parallel.
set -e
cd "$(dirname "$0")/.."; export OMP_NUM_THREADS=1; export PYTHONPATH=code; mkdir -p logs
# ---- 2-d pendulum (o1_train: N0 = 400k, T = 1, H = 128, Adam + EMA; seeds fixed) ----
for job in "gauss 0.1 eps" "gauss 0.1 score" "gauss 0.3 eps" "gauss 0.3 score" "fan 0.1 eps" "fan 0.1 score" "fan 0.3 eps" "fan 0.3 score" "needle 0.1 eps" "needle 0.3 eps"; do
  python3 code/o1_train.py $job 300000 1 >> logs/retrain_2d.txt 2>&1
done
python3 code/o1_train.py gauss 0.03 eps 400000 1 256 >> logs/retrain_2d.txt 2>&1                          # stage P(b): H = 256 (reference only)
O1_BATCH=4096 python3 code/o1_train.py gauss 0.03 eps 150000 1 >> logs/retrain_2d.txt 2>&1                 # stage Q(d)
O1_BATCH=16384 python3 code/o1_train.py gauss 0.03 eps 75000 1 >> logs/retrain_2d.txt 2>&1                 # stage R1
# ---- 4-d coupled pendulums (p1_train: N0 = 200k, T = 0.5, H = 128) ----
for e in 0.1 0.3; do for s in 0.1 0.3; do for w in eps score; do python3 code/p1_train.py $e $s $w 300000 1 >> logs/retrain_4d.txt 2>&1; done; done; done
for seed in 2 3; do python3 code/p1_train.py 0.3 0.1 eps 300000 $seed >> logs/retrain_4d.txt 2>&1; done
# ---- comparisons with the exact score (kernel sums over flow-carried quadrature nodes) ----
O1_NCMP=200000 python3 code/o1_compare.py gauss 0.1 eps 1 results/o1_compare.json >> logs/recompare.txt 2>&1     # stage O used 200k samples for this one entry
O1_NCMP=100000 python3 code/o1_compare.py gauss 0.1 score 1 results/o1_compare.json >> logs/recompare.txt 2>&1
for st in gauss fan; do for s in 0.1 0.3; do [ "$st $s" = "gauss 0.1" ] && continue; O1_NCMP=100000 python3 code/o1_compare.py $st $s eps,score 1 results/o1_compare.json >> logs/recompare.txt 2>&1; done; done
O1_NCMP=100000 python3 code/o1_compare.py needle 0.1 eps 1 results/p2_compare.json >> logs/recompare.txt 2>&1
O1_NCMP=100000 python3 code/o1_compare.py needle 0.3 eps 1 results/r3_needle_compare.json >> logs/recompare.txt 2>&1
O1_BATCH=16384 O1_NCMP=100000 python3 code/o1_compare.py gauss 0.03 eps 1 results/r1_compare.json >> logs/recompare.txt 2>&1
for e in 0.1 0.3; do for s in 0.1 0.3; do python3 code/p1_compare.py $e $s eps,score 1 >> logs/recompare.txt 2>&1; done; done
for seed in 2 3; do P1_CMPFILE=results/q1_seeds.json python3 code/p1_compare.py 0.3 0.1 eps $seed >> logs/recompare.txt 2>&1; done
# ---- boundary (I) in 4-d with the learned score (stage R2) ----
for s in 0.1 0.3; do python3 code/r2_boundary4d.py $s learned >> logs/recompare.txt 2>&1; done
