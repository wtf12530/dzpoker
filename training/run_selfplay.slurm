#!/bin/bash
#SBATCH --job-name=poker_selfplay
#SBATCH --output=slurm-selfplay-%j.out
#SBATCH --error=slurm-selfplay-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=8G

# Example SLURM script to run multiple data collection workers using srun.
# Adjust --ntasks, --cpus-per-task, and module loads to your cluster environment.

WORKDIR=$HOME/dzpoker
cd $WORKDIR
source .venv/bin/activate

# Launch 4 parallel workers collecting selfplay data into separate files
for i in 0 1 2 3; do
  srun --exclusive -N1 -n1 python training/collect_selfplay.py --game limit-holdem --num-games 500 --out training/data/selfplay_worker_${i}.npz &
done
wait

# Optionally merge the generated .npz files into one dataset (left as an exercise)
