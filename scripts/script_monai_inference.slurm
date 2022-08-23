#!/bin/bash
#SBATCH -C rome
#SBATCH --nodes=1
#SBATCH -N1 --ntasks-per-node=1 -o /mnt/home/pgrover/3dunet/logs/log_monai_inference.%j.out -e /mnt/home/pgrover/3dunet/logs/log_monai_inference.%j.err

module load gcc/10
module load python/3
module load cuda
cd /mnt/home/pgrover/3dunet/
python monai_inference.py