### an example for training Naiver-Stokes equation on irregular domains
python train.py --gpu 0 --use-normalizer unit  --normalize_x unit --component all --comment rel2  --loss-name rel2 --epochs 500 --batch-size 10 --model-name GNOT --optimizer AdamW --weight-decay 0.00005   --lr 0.001 --lr-method cycle  --grad-clip 1000.0   --n-hidden 128 --n-layers 3  --use-tb 0  # 2>&1 & sleep 20s
