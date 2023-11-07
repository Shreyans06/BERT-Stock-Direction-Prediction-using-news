#!/bin/bash
#SBATCH -J amazon_train_daily  ## Job name
#SBATCH -c 8
#SBATCH -p general
#SBATCH -t 1-00:00:00
#SBATCH -G a100:1
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --export=NONE

module load mamba/latest

source activate finbert

python dataset.py --data_dir ./dataset/amazon_news_daily_summarized_spacy.csv --batch_size 16 --token_len 128 --model ProsusAI/finbert --lr 1e-5