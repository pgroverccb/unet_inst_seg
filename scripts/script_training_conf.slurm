#!/bin/bash
#SBATCH --nodes=2
#SBATCH -N2 --ntasks-per-node=8 -o /mnt/home/pgrover/3dunet/logs/log_training_conf.%j.out -e /mnt/home/pgrover/continous_cell_cycle_stage_pred/logs/log_training_conf.%j.err

module load gcc/10
module load python/3
cd /mnt/home/pgrover/3dunet/
python training_conf.py
