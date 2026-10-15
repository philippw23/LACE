#!/bin/bash
#SBATCH --job-name=lace_gradcam
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/lace/slurm-%j_lace_gradcam.out"

home_dir="/mnt/nfs/homedirs/$USER"
export HOME=$home_dir
cd ${SLURM_SUBMIT_DIR}
echo Starting job ${SLURM_JOBID}
echo SLURM assigned me these nodes:
squeue -j ${SLURM_JOBID} -O nodelist | tail -n +2

MY_CONDA_ENV="master"
export SSL_CERT_FILE=$home_dir/miniconda3/envs/$MY_CONDA_ENV/ssl/cert.pem
export CONDA_EXE=$home_dir/miniconda3/bin/conda
source $home_dir/miniconda3/etc/profile.d/conda.sh
conda activate $MY_CONDA_ENV
echo Environment activated

export HF_HOME=$home_dir/.cache/huggingface
export TRANSFORMERS_CACHE=$home_dir/.cache/huggingface/transformers
export PATH=$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin:$PATH
export PYTHONPATH=$home_dir/Project/src

# All architecture hyperparameters + the backbone path + train age-stats are read
# from the head checkpoint itself. With no --image_manifest, the BTXRD manifest
# recorded in that checkpoint's own config is used (every LACE CV fold is scored
# on it), so this defaults to visualizing BTXRD images.
#
HEAD_CHECKPOINT=$home_dir/Project/results/lace_v2_downstream/best_20261007_080057_dnemrr3j.pt

N_PER_CLASS=8
TARGET=pred   # pred | benign | malignant
OUT_DIR=$home_dir/Project/results/lace_v2_gradcam/dnemrr3j
# ─────────────────────────────────────────────────────────────────────────────

python $home_dir/Project/src/LACE/train/gradcam.py \
    --head_checkpoint $HEAD_CHECKPOINT \
    --n_per_class     $N_PER_CLASS \
    --target          $TARGET \
    --out_dir         $OUT_DIR
