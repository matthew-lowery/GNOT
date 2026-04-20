#!/bin/bash


for dataset in "flow_cylinder_shedding_2d" "flow_cylinder_laminar_2d"  "taylor_green_numerical_2d"  "taylor_green_exact_2d"  "backward_facing_step_2d"  "buoyancy_cavity_flow_2d"  "lid_cavity_flow_2d"  "merge_vortices_2d"; do
for ne in 16 32 64 128; do
python3 -u train.py --comment rel2 --component 'reduce-all' --use-normalizer unit --normalize_x unit --epochs 500 --batch-size 20 --npoints=1000 --optimizer AdamW --n-hidden=$ne --dataset=$dataset --n-layers=3
done
for d in 3 4 5; do
python3 -u train.py --comment rel2 --component 'reduce-all' --use-normalizer unit --normalize_x unit --epochs 500 --batch-size 20 --npoints=1000 --optimizer AdamW --n-hidden=16 --dataset=$dataset --n-layers=$d
done
for nh in 2 4; do
python3 -u train.py --comment rel2 --component 'reduce-all' --use-normalizer unit --normalize_x unit --epochs 500 --batch-size 20 --npoints=1000 --optimizer AdamW --n-hidden=16 --dataset=$dataset --n-layers=3 --n-head=$nh
done
done