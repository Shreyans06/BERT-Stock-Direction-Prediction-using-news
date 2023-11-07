#!/bin/bash
#SBATCH -J apple_summary  ## Job name
#SBATCH -c 12
#SBATCH -p general
#SBATCH -t 1-00:00:00
#SBATCH -G a100:1
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --export=NONE

module load mamba/latest

source activate finbert

python dataset.py --data_dir ./dataset/apple_news_daily.csv --batch_size 32 --token_len 128 --summarize True 