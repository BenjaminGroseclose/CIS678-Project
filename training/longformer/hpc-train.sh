#!/bin/bash

#SBATCH --gpus-per-node=2 ## nb of GPU(s)

#SBATCH --mem=8192 ##Memory I want to use in MB

#SBATCH --time=20:25:00 ## time it will take to complete job

#SBATCH --partition=class ##Partition I want to use

#SBATCH --ntasks=1 ##Number of task

#SBATCH --job-name=text-detection-v1 ## Name of job

#SBATCH --output=text-detection-v1-.%j.out ##Name of output file

plm_dir="allenai/longformer-base-4096"
seed=42629309
data_path="../data/cross_domains_cross_models"
train_file="$data_path/train.csv"
valid_file="$data_path/valid.csv"
out_dir="./output_samples_${seed}_lfbase"
time=$(date +'%m:%d:%H:%M')
mkdir -p $out_dir

source /mnt/home/groseclb/CIS678-Project/venv./bin/activate

python /mnt/home/groseclb/CIS678-Project/training/longformer/main.py

deactivate

