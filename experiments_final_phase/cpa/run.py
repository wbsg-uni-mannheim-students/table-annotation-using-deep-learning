import os

cmd = """CUDA_VISIBLE_DEVICES=2 python train_last_phase.py --method cpa \
    --serialization neighbor \
    --model_name roberta-base \
    --max_length 512 \
    --window_size 5 \
    --aug None \
    --batch_size 32 \
    --epoch 30 \
    --random_seed 0 \
    --preprocess None \
    --lr 5e-5 \
    --mp 0.5"""
print(cmd)
os.system(cmd)