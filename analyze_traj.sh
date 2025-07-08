#!/bin/bash
#SBATCH --partition=main
# SBATCH --partition=largemem
#SBATCH --job-name="analyze_traj"
# Default rohs_108 or katritch_223
#SBATCH --account=katritch_223
#SBATCH --nodes=1
# SBATCH --ntasks=64
# SBATCH --ntasks-per-node=64
# SBATCH --constraint=epyc-7513
#SBATCH --time=20:00:00
#SBATCH --mem=0
#SBATCH --mail-user=mynguyen@usc.edu
#SBATCH --mail-type=all
#SBATCH --exclusive

export PATH=/home1/mynguyen/.conda/envs/py310/bin:$PATH
GOTOFOLDERS=() #trajcat_Set_0_0_0 trajcat_Set_0_0_1 trajcat_Set_0_0_2 trajcat_Set_0_0_3 trajcat_Set_0_0_4)

cd ./Trajectories
Main_DIR=$(pwd)

for pawn in ${GOTOFOLDERS[@]}
do
 # check if the directory exists
 if [ ! -d "${pawn}" ]; then
   continue
 fi
 echo "====== $pawn ======"
 cd ${pawn}
 python mdtraj_contact.py > mdtraj_contact.log
 cd ${Main_DIR}
done

cd rmsd_analysis
python md_rms.py > md_rms.log 
echo "====== Finished ======"
