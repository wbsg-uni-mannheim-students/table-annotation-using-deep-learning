#!/bin/bash
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=50:00:00
#SBATCH --mem=50gb
#SBATCH --export=ALL
#SBATCH --output=./slurm_output/%j.out
#SBATCH --mail-user=rengchizder@outlook.com
#SBATCH --mail-type=ALL
SERIALIZATION=$1
MAXLEN=$2
WINDOWSIZE=$3
PREPROCESS=$4
SEED=$5
AUG=$6
LR=$7
MP=$8
python train_last_phase.py \
    --method cta \
    --serialization $SERIALIZATION \
    --model_name roberta-base \
    --max_length $MAXLEN \
    --window_size $WINDOWSIZE \
    --aug $AUG \
    --batch_size 32 \
    --epoch 30 \
    --random_seed $SEED \
    --preprocess $PREPROCESS \
    --lr $LR \
    --mp $MP