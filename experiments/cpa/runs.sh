#!/bin/bash
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --ntasks-per-node=32
#SBATCH --mem=50gb
#SBATCH --export=ALL
#SBATCH --output=./slurm_output/%j.out
SERIALIZATION=$1
MAXLEN=$2
WINDOWSIZE=$3
PREPROCESS=$4
SEED=$5
AUG=$6
LR=$7
python train_lm_final.py \
    --method cpa \
    --serialization $SERIALIZATION \
    --model_name bert-base-uncased \
    --max_length $MAXLEN \
    --window_size $WINDOWSIZE \
    --aug $AUG \
    --batch_size 16 \
    --epoch 20 \
    --random_seed $SEED \
    --preprocess $PREPROCESS \
    --lr $LR
