#!/bin/bash
# sbatch runs.sh single-column 512 0 None 0 None 5e-5
# sbatch runs.sh single-column 512 0 None 1 None 5e-5
# sbatch runs.sh single-column 512 0 None 2 None 5e-5
# sbatch runs.sh single-column 512 0 tuta 0 None 5e-5
# sbatch runs.sh single-column 512 0 tuta 1 None 5e-5
# sbatch runs.sh single-column 512 0 tuta 2 None 5e-5
# sbatch runs.sh single-column 512 0 median 0 None 5e-5
# sbatch runs.sh single-column 512 0 median 1 None 5e-5
# sbatch runs.sh single-column 512 0 median 2 None 5e-5
# sbatch runs.sh single-column 512 0 mean 0 None 5e-5
# sbatch runs.sh single-column 512 0 mean 1 None 5e-5
# # sbatch runs.sh single-column 512 0 mean 2 None 5e-5
# sbatch runs.sh single-column 512 0 None 0 delete_random_cell 5e-5
# sbatch runs.sh single-column 512 0 None 1 delete_random_cell 5e-5
# sbatch runs.sh single-column 512 0 None 2 delete_random_cell 5e-5

# sbatch runs.sh neighbor 512 1 None 0 None 5e-5 # 512 / (main_col + other+cols + spcial_tokens)
# sbatch runs.sh neighbor 512 1 None 1 None 5e-5
# sbatch runs.sh neighbor 512 1 None 2 None 5e-5
# sbatch runs.sh neighbor 512 1 tuta 0 None 5e-5
# sbatch runs.sh neighbor 512 1 tuta 1 None 5e-5
# sbatch runs.sh neighbor 512 1 tuta 2 None 5e-5
# sbatch runs.sh neighbor 512 1 median 0 None 5e-5
# sbatch runs.sh neighbor 512 1 median 1 None 5e-5
# sbatch runs.sh neighbor 512 1 median 2 None 5e-5
# sbatch runs.sh neighbor 512 1 mean 0 None 5e-5
# sbatch runs.sh neighbor 512 1 mean 1 None 5e-5
# sbatch runs.sh neighbor 512 1 mean 2 None 5e-5
# sbatch runs.sh neighbor 512 2 None 0 None 5e-5
# sbatch runs.sh neighbor 512 2 None 1 None 5e-5
# sbatch runs.sh neighbor 512 2 None 2 None 5e-5
# sbatch runs.sh neighbor 512 2 tuta 0 None 5e-5
# sbatch runs.sh neighbor 512 2 tuta 1 None 5e-5
# sbatch runs.sh neighbor 512 2 tuta 2 None 5e-5
# sbatch runs.sh neighbor 512 2 median 0 None 5e-5 
# sbatch runs.sh neighbor 512 2 median 1 None 5e-5
# sbatch runs.sh neighbor 512 2 median 2 None 5e-5
# sbatch runs.sh neighbor 512 2 mean 0 None 5e-5 
# sbatch runs.sh neighbor 512 2 mean 1 None 5e-5
# sbatch runs.sh neighbor 512 2 mean 2 None 5e-5

# sbatch runs.sh random_neighbor 512 1 None 0 None 5e-5 # 512 / (main_col + other+cols + spcial_tokens)
# sbatch runs.sh random_neighbor 512 1 None 1 None 5e-5
# sbatch runs.sh random_neighbor 512 1 None 2 None 5e-5
# sbatch runs.sh random_neighbor 512 1 tuta 0 None 5e-5
# sbatch runs.sh random_neighbor 512 1 tuta 1 None 5e-5
# sbatch runs.sh random_neighbor 512 1 tuta 2 None 5e-5
# sbatch runs.sh random_neighbor 512 1 median 0 None 5e-5 
# sbatch runs.sh random_neighbor 512 1 median 1 None 5e-5
# sbatch runs.sh random_neighbor 512 1 median 2 None 5e-5
# sbatch runs.sh random_neighbor 512 1 mean 0 None 5e-5 
# sbatch runs.sh random_neighbor 512 1 mean 1 None 5e-5
# sbatch runs.sh random_neighbor 512 1 mean 2 None 5e-5
# sbatch runs.sh random_neighbor 512 2 None 0 None 5e-5
# sbatch runs.sh random_neighbor 512 2 None 1 None 5e-5
# sbatch runs.sh random_neighbor 512 2 None 2 None 5e-5
# sbatch runs.sh random_neighbor 512 2 tuta 0 None 5e-5
# sbatch runs.sh random_neighbor 512 2 tuta 1 None 5e-5
# sbatch runs.sh random_neighbor 512 2 tuta 2 None 5e-5
# sbatch runs.sh random_neighbor 512 2 median 0 None 5e-5
# sbatch runs.sh random_neighbor 512 2 median 1 None 5e-5
# sbatch runs.sh random_neighbor 512 2 median 2 None 5e-5
# sbatch runs.sh random_neighbor 512 2 mean 0 None 5e-5
# sbatch runs.sh random_neighbor 512 2 mean 1 None 5e-5
# sbatch runs.sh random_neighbor 512 2 mean 2 None 5e-5

# sbatch runs.sh tabert 512 0 None 0 None 5e-5
# sbatch runs.sh tabert 512 0 None 1 None 5e-5
# sbatch runs.sh tabert 512 0 None 2 None 5e-5
# sbatch runs.sh tabert 512 0 tuta 0 None 5e-5
# sbatch runs.sh tabert 512 0 tuta 1 None 5e-5
# sbatch runs.sh tabert 512 0 tuta 2 None 5e-5
# sbatch runs.sh tabert 512 0 median 0 None 5e-5
# sbatch runs.sh tabert 512 0 median 1 None 5e-5
# sbatch runs.sh tabert 512 0 median 2 None 5e-5
# sbatch runs.sh tabert 512 0 mean 0 None 5e-5
# sbatch runs.sh tabert 512 0 mean 1 None 5e-5
# sbatch runs.sh tabert 512 0 mean 2 None 5e-5

# sbatch runs.sh summary 512 0 None 0 None 5e-5 # 512 / (main_col + other+cols + spcial_tokens)
# sbatch runs.sh summary 512 0 None 1 None 5e-5
# sbatch runs.sh summary 512 0 None 2 None 5e-5
# sbatch runs.sh summary 512 0 tuta 0 None 5e-5
# sbatch runs.sh summary 512 0 tuta 1 None 5e-5
# sbatch runs.sh summary 512 0 tuta 2 None 5e-5
# sbatch runs.sh summary 512 0 median 0 None 5e-5
# sbatch runs.sh summary 512 0 median 1 None 5e-5
# sbatch runs.sh summary 512 0 median 2 None 5e-5
# sbatch runs.sh summary 512 0 mean 0 None 5e-5
# sbatch runs.sh summary 512 0 mean 1 None 5e-5
# sbatch runs.sh summary 512 0 mean 2 None 5e-5

sbatch runs.sh freq 512 0 None 0 None 5e-5 # 512 / (main_col + other+cols + spcial_tokens)
sbatch runs.sh freq 512 0 None 1 None 5e-5
sbatch runs.sh freq 512 0 None 2 None 5e-5
sbatch runs.sh freq 512 0 tuta 0 None 5e-5
sbatch runs.sh freq 512 0 tuta 1 None 5e-5
sbatch runs.sh freq 512 0 tuta 2 None 5e-5
sbatch runs.sh freq 512 0 median 0 None 5e-5
sbatch runs.sh freq 512 0 median 1 None 5e-5
sbatch runs.sh freq 512 0 median 2 None 5e-5
sbatch runs.sh freq 512 0 mean 0 None 5e-5
sbatch runs.sh freq 512 0 mean 1 None 5e-5
sbatch runs.sh freq 512 0 mean 2 None 5e-5

# SERIALIZATION=$1
# MAXLEN=$2
# WINDOWSIZE=$3
# PREPROCESS=$4
# SEED=$5
# AUG=$6
# LR=$7