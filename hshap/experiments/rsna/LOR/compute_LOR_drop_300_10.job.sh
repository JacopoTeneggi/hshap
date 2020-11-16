#!/bin/sh
#SBATCH --job-name=LOR_array_300_10
#SBATCH --partition=gpuk80
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --array=100,200,400,800,1600
#SBATCH --mail-type=end
#SBATCH --mail-user=jtenegg1@jhu.edu
#SBATCH --time=24:0:0

module load python/3.7
module load cuda/10.1
python -m pip install -e /home-2/jtenegg1@jhu.edu/work/repo/hshap/

python /home-2/jtenegg1@jhu.edu/work/repo/hshap/hshap/experiments/rsna/LOR/compute_LOR_drop.py "/home-2/jtenegg1@jhu.edu/work" 300 $SLURM_ARRAY_TASK_ID 10