#!/bin/bash
#
#SBATCH --job-name=nqp_exam
#SBATCH --comment="nqp exam: exact diagonalization of spin wheel"
#SBATCH --ntasks=50
#SBATCH --partition=cip
#SBATCH --mem-per-cpu=2048
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yudong.Sun@physik.uni-muenchen.de
#SBATCH --chdir=/home/y/Yudong.Sun/0_modules/NQP/nqp-exam/src
#SBATCH --output=/home/y/Yudong.Sun/0_modules/NQP/nqp-exam/slurm/slurm.%j.%N.out
#SBATCH --error=/home/y/Yudong.Sun/0_modules/NQP/nqp-exam/slurm/slurm.%j.%N.err.out

source /project/cip/2023-SS-NQP/init_modules.sh
module unload python
module load python/3.10-2022.08
python3 main.py
