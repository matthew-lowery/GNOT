#!/bin/bash


sp() { 
    local pycmd="$1"
    local hr="${2:-1}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgcs-delta-gpu
#SBATCH --job-name=myjob
#SBATCH --time=${hr}:00:00
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --output=./k/%x_%A.out
#SBATCH --error=./k/%x_%A.err

module purge
export PATH=/u/mlowery/.conda/envs/gnot/bin:\$PATH
cd /u/mlowery/GNOT/
$pycmd
EOF
}

seed=1

# lowest grids: torus = 2400; sphere = 2562
# fixed middle values: n-head = 4; n-hidden = 64; n-layers = 6

for nl in 4 6 8; do
sp "python3 -u train.py --dataset=poisson_sphere_3d --seed=$seed --component all --use-normalizer unit --normalize_x unit --epochs 500 --batch-size 1 --val-batch-size 1 --model-name GNOT --optimizer AdamW --weight-decay 0.000005 --lr 0.001 --lr-method cycle --grad-clip 1000.0 --comment manifold_tune --wandb --train-num=1000 --npoints=2562 --n-head=4 --n-hidden=64 --n-layers=$nl" 12
sp "python3 -u train.py --dataset=poisson_torus_3d --seed=$seed --component all --use-normalizer unit --normalize_x unit --epochs 500 --batch-size 1 --val-batch-size 1 --model-name GNOT --optimizer AdamW --weight-decay 0.000005 --lr 0.001 --lr-method cycle --grad-clip 1000.0 --comment manifold_tune --wandb --train-num=1000 --npoints=2400 --n-head=4 --n-hidden=64 --n-layers=$nl" 12
sp "python3 -u train.py --dataset=nlpoisson_sphere_3d --seed=$seed --component all --use-normalizer unit --normalize_x unit --epochs 500 --batch-size 1 --val-batch-size 1 --model-name GNOT --optimizer AdamW --weight-decay 0.000005 --lr 0.001 --lr-method cycle --grad-clip 1000.0 --comment manifold_tune --wandb --train-num=1000 --npoints=2562 --n-head=4 --n-hidden=64 --n-layers=$nl" 12
sp "python3 -u train.py --dataset=nlpoisson_torus_3d --seed=$seed --component all --use-normalizer unit --normalize_x unit --epochs 500 --batch-size 1 --val-batch-size 1 --model-name GNOT --optimizer AdamW --weight-decay 0.000005 --lr 0.001 --lr-method cycle --grad-clip 1000.0 --comment manifold_tune --wandb --train-num=1000 --npoints=2400 --n-head=4 --n-hidden=64 --n-layers=$nl" 12
sp "python3 -u train.py --dataset=ADRSHEAR_3d --seed=$seed --component all --use-normalizer unit --normalize_x unit --epochs 500 --batch-size 1 --val-batch-size 1 --model-name GNOT --optimizer AdamW --weight-decay 0.000005 --lr 0.001 --lr-method cycle --grad-clip 1000.0 --comment manifold_tune --wandb --train-num=2000 --n-head=4 --n-hidden=64 --n-layers=$nl" 20
done

for nh in 2 4 6 8; do
sp "python3 -u train.py --dataset=poisson_sphere_3d --seed=$seed --component all --use-normalizer unit --normalize_x unit --epochs 500 --batch-size 1 --val-batch-size 1 --model-name GNOT --optimizer AdamW --weight-decay 0.000005 --lr 0.001 --lr-method cycle --grad-clip 1000.0 --comment manifold_tune --wandb --train-num=1000 --npoints=2562 --n-head=$nh --n-hidden=64 --n-layers=6" 12
sp "python3 -u train.py --dataset=poisson_torus_3d --seed=$seed --component all --use-normalizer unit --normalize_x unit --epochs 500 --batch-size 1 --val-batch-size 1 --model-name GNOT --optimizer AdamW --weight-decay 0.000005 --lr 0.001 --lr-method cycle --grad-clip 1000.0 --comment manifold_tune --wandb --train-num=1000 --npoints=2400 --n-head=$nh --n-hidden=64 --n-layers=6" 12
sp "python3 -u train.py --dataset=nlpoisson_sphere_3d --seed=$seed --component all --use-normalizer unit --normalize_x unit --epochs 500 --batch-size 1 --val-batch-size 1 --model-name GNOT --optimizer AdamW --weight-decay 0.000005 --lr 0.001 --lr-method cycle --grad-clip 1000.0 --comment manifold_tune --wandb --train-num=1000 --npoints=2562 --n-head=$nh --n-hidden=64 --n-layers=6" 12
sp "python3 -u train.py --dataset=nlpoisson_torus_3d --seed=$seed --component all --use-normalizer unit --normalize_x unit --epochs 500 --batch-size 1 --val-batch-size 1 --model-name GNOT --optimizer AdamW --weight-decay 0.000005 --lr 0.001 --lr-method cycle --grad-clip 1000.0 --comment manifold_tune --wandb --train-num=1000 --npoints=2400 --n-head=$nh --n-hidden=64 --n-layers=6" 12
sp "python3 -u train.py --dataset=ADRSHEAR_3d --seed=$seed --component all --use-normalizer unit --normalize_x unit --epochs 500 --batch-size 1 --val-batch-size 1 --model-name GNOT --optimizer AdamW --weight-decay 0.000005 --lr 0.001 --lr-method cycle --grad-clip 1000.0 --comment manifold_tune --wandb --train-num=2000 --n-head=$nh --n-hidden=64 --n-layers=6" 20
done

for ne in 32 64 128; do
sp "python3 -u train.py --dataset=poisson_sphere_3d --seed=$seed --component all --use-normalizer unit --normalize_x unit --epochs 500 --batch-size 1 --val-batch-size 1 --model-name GNOT --optimizer AdamW --weight-decay 0.000005 --lr 0.001 --lr-method cycle --grad-clip 1000.0 --comment manifold_tune --wandb --train-num=1000 --npoints=2562 --n-head=4 --n-hidden=$ne --n-layers=6" 12
sp "python3 -u train.py --dataset=poisson_torus_3d --seed=$seed --component all --use-normalizer unit --normalize_x unit --epochs 500 --batch-size 1 --val-batch-size 1 --model-name GNOT --optimizer AdamW --weight-decay 0.000005 --lr 0.001 --lr-method cycle --grad-clip 1000.0 --comment manifold_tune --wandb --train-num=1000 --npoints=2400 --n-head=4 --n-hidden=$ne --n-layers=6" 12
sp "python3 -u train.py --dataset=nlpoisson_sphere_3d --seed=$seed --component all --use-normalizer unit --normalize_x unit --epochs 500 --batch-size 1 --val-batch-size 1 --model-name GNOT --optimizer AdamW --weight-decay 0.000005 --lr 0.001 --lr-method cycle --grad-clip 1000.0 --comment manifold_tune --wandb --train-num=1000 --npoints=2562 --n-head=4 --n-hidden=$ne --n-layers=6" 12
sp "python3 -u train.py --dataset=nlpoisson_torus_3d --seed=$seed --component all --use-normalizer unit --normalize_x unit --epochs 500 --batch-size 1 --val-batch-size 1 --model-name GNOT --optimizer AdamW --weight-decay 0.000005 --lr 0.001 --lr-method cycle --grad-clip 1000.0 --comment manifold_tune --wandb --train-num=1000 --npoints=2400 --n-head=4 --n-hidden=$ne --n-layers=6" 12
sp "python3 -u train.py --dataset=ADRSHEAR_3d --seed=$seed --component all --use-normalizer unit --normalize_x unit --epochs 500 --batch-size 1 --val-batch-size 1 --model-name GNOT --optimizer AdamW --weight-decay 0.000005 --lr 0.001 --lr-method cycle --grad-clip 1000.0 --comment manifold_tune --wandb --train-num=2000 --n-head=4 --n-hidden=$ne --n-layers=6" 20
done
