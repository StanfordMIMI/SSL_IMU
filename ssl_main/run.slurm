#!/bin/bash

#SBATCH -n 1
#SBATCH --array=0-2

dsets=(
    movi
    amass
    combined
)

srun python a1_ssl.py --ssldset "${dsets[$SLURM_ARRAY_TASK_ID]}"