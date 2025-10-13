#!/bin/bash


sp() {
    local pycmd="$1"
    local hr="${2:-8}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpuA100x4
#SBATCH --account=bduo-delta-gpu
#SBATCH --job-name=myjob
#SBATCH --time=18:00:00
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --output=./trib/%x_%A.out
#SBATCH --error=./trib/%x_%A.err

module purge
export PATH=/u/mlowery/.conda/envs/gnot/bin:\$PATH
cd /u/mlowery/GNOT/
$pycmd
EOF
}

dataset='flow_cylinder_shedding_2d'
for nl in 3; do
for ne in 32; do
for at in linear; do
for seed in 1; do
sp "python3 -u train.py --npoints=1000 --attn-type=$at --use-normalizer unit --normalize_x unit --epochs 10000 --batch-size 100 --optimizer AdamW --seed=$seed --n-hidden=$ne --dataset=$dataset --n-layers=$nl"
done
done
done
done
