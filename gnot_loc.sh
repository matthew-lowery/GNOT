#!/bin/bash
dataset='merge_vortices_2d'
for nl in 3; do
for ne in 32; do
for at in linear; do
for seed in 1; do
python3 -u train.py --npoints='all' --attn-type=$at --use-normalizer unit --normalize_x unit --epochs 10000 --batch-size 100 --npoints=1000 --optimizer AdamW --seed=$seed --n-hidden=$ne --dataset=$dataset --n-layers=$nl
done
done
done
done
