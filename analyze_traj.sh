#!/bin/bash
#SBATCH --account=katritch_502
#SBATCH --partition=epyc-64

# SBATCH --partition=main
# SBATCH --partition=largemem
#SBATCH --job-name="analyze_traj"
# Default rohs_108 or katritch_223
# SBATCH --account=katritch_223
#SBATCH --nodes=1
# SBATCH --ntasks=64
# SBATCH --ntasks-per-node=64
# SBATCH --constraint=epyc-7513
#SBATCH --time=2:00:00
#SBATCH --mem=0
#SBATCH --mail-user=mynguyen@usc.edu
#SBATCH --mail-type=all
#SBATCH --exclusive

export PATH=/home1/mynguyen/.conda/envs/py310/bin:$PATH

# python md_rms.py > md_rms.log 
python md_orientation.py > md_orient.log 
python md_contact.py > md_contact.log 
echo "====== Finished ======"
