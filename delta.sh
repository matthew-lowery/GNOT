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
#SBATCH --partition=gpuH200x8,gpuA100x4,gpuA100x8,gpuA40x4
#SBATCH --account=bduo-delta-gpu
#SBATCH --job-name=myjob
#SBATCH --time=24:00:00
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --output=./diffr2/%x_%A.out
#SBATCH --error=./diffr2/%x_%A.err

module purge
export PATH=/u/mlowery/.conda/envs/gnot/bin:\$PATH
cd /u/mlowery/GNOT/
$pycmd
EOF
}

for dataset in 
for nl in 2 3 4 5; do
sp "python -u train.py --gpu 0 --attn-type=$at --use-normalizer unit  --normalize_x unit --component all --comment rel2  --loss-name rel2 --epochs 1000 --val-batch-size 100 --batch-size 100 --model-name GNOT --optimizer AdamW --weight-decay 0.00005   --lr 0.001 --lr-method cycle  --grad-clip 1000.0 --seed=$seed  --n-hidden=$ne --dataset=diffr3d --n-layers=$nl  --use-tb 0  # 2>&1 & sleep 20s"
done
for ne in 64; do
for at in linear; do
for seed in 1 2 3 4; do

done
done
done
done
