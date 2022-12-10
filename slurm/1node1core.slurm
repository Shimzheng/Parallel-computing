#!/bin/bash


#SBATCH --partition=snowy
#SBATCH --job-name="1node1core"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-user=wouyang@unimelb.edu.au
#SBATCH --mail-type=ALL
#SBATCH --time=1-0:00:00

# Load required modules
module --force purge
module load mpi4py/3.0.2-timed-pingpong
module load foss/2019b 
module load python/3.7.4 


#Launch multiple process python code
echo "1 node 1 core"
time srun -n 1 python3 master.py -data /data/projects/COMP90024/bigTwitter.json -grid /data/projects/COMP90024/sydGrid.json -code langCode.json
