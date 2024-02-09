#!/bin/bash
#SBATCH --error=log_scripts/finetune_for_se.err
#SBATCH --output=log_scripts/finetune_for_se.out
#SBATCH -J
### #SBATCH -p slim18
#SBATCH --ntasks=16  
#SBATCH --mem=60G
#SBATCH -t 0


bash ~/.bashrc
source /home/haehnel/miniconda3/etc/profile.d/conda.sh
conda activate py3.10

python Sentence_Entailment_BERT5G.py
