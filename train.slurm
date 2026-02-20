#!/bin/bash
#SBATCH --job-name=diffusion2
#SBATCH --constraint=v100-32g
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --qos=qos_gpu-t3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --output=out.out
#SBATCH --error=logs.out
#SBATCH -A ehe@v100
module purge # nettoyer les modules herites par defaut
conda deactivate # desactiver les environnements herites par defaut
module load python
module load arch/a100 # selectionner les modules compiles pour les A100
source /lustre/fsn1/projects/rech/ehe/udq27fb/schrodinger-ia2hl/.venv/bin/activate
export PYTHONPATH=/lustre/fsn1/projects/rech/ehe/udq27fb/schrodinger-ia2hl/.venv/lib/python3.9/site-packages:$PYTHONPATH
set -x # activer l’echo des commandes
srun python train.py # executer son script
