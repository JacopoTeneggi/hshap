#SBATCH --job-name=LOR_array
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --array=100,200,400,800,1600
#SBATCH --mail-type=end
#SBATCH --mail-user=jtenegg1@jhu.edu
#SBATCH --time=24:0:0

module load python/3.7
python -m pip install -e /home-2/jtenegg1@jhu.edu/work/repo/hshap/

python ./compute_LOR_drop.py "/home-2/jtenegg1@jhu.edu/work" 300 $SLURM_ARRAY_TASK_ID 10