#!/bin/bash


for dataset in "flow_cylinder_shedding" "flow_cylinder_laminar" "taylor_green_numerical" "taylor_green_exact" "backward_facing_step" "buoyancy_cavity_flow" "lid_cavity_flow" "merge_vortices"; do
for ne in 16 32 64 128; do
python3 -u train.py --comment rel2 --component 'all-reduce' --use-normalizer unit --normalize_x unit --epochs 500 --batch-size 20 --npoints=1000 --optimizer AdamW --n-hidden=$ne --dataset=$dataset --n-layers=3
done
for d in 3 4 5; do
python3 -u train.py --comment rel2 --component 'all-reduce' --use-normalizer unit --normalize_x unit --epochs 500 --batch-size 20 --npoints=1000 --optimizer AdamW --n-hidden=16 --dataset=$dataset --n-layers=$d
done
for nh in 2 4; do
python3 -u train.py --comment rel2 --component 'all-reduce' --use-normalizer unit --normalize_x unit --epochs 500 --batch-size 20 --npoints=1000 --optimizer AdamW --n-hidden=16 --dataset=$dataset --n-layers=3 --n-head=$nh
done
done
