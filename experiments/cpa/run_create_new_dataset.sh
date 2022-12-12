#!/bin/bash
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --ntasks-per-node=32
#SBATCH --mem=50gb
#SBATCH --export=ALL
python create_new_dataset.py