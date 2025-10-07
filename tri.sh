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

for nl in 5; do
for ne in 64; do
for at in linear; do
for seed in 1 2 3 4; do
sp "python3 -u train.py --gpu 0 --attn-type=$at --use-normalizer unit  --normalize_x unit --component all --comment rel2  --loss-name rel2 --epochs 10000 --batch-size 100 --model-name GNOT --optimizer AdamW --weight-decay 0.00005   --lr 0.001 --lr-method cycle  --grad-clip 1000.0 --seed=$seed  --n-hidden=$ne --dataset=tri2d --n-layers=$nl  --use-tb 0  # 2>&1 & sleep 20s"
done
done
done
done
