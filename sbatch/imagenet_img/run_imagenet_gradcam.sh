#!/bin/bash
#SBATCH --job-name=imagenet_gradcam
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/slurm-%j_imagenet_gradcam.out"

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

# All hyperparameters + train age-stats are read from the head checkpoint itself.
# With no --image_manifest, the BTXRD manifest recorded in that checkpoint's own
# config is used — same default as LACE's run_lace_gradcam.sh, so pass the same
# --patids to both for an image-for-image comparison.
#
# This is the fold0 CV checkpoint from slurm-7899 (same CV split pool as LACE's
# A4 run, but frozen-encoder mode — no per-fold backbone checkpoint).
FOLD=0   # <-- set this (0-3 currently on disk; re-run the CV sweep for more)
HEAD_CHECKPOINT=$home_dir/Project/results/imagenet_img/cv_fold${FOLD}_split_binary_fold${FOLD}_7899/best_checkpoint.pt

N_PER_CLASS=8
TARGET=pred   # pred | benign | malignant
OUT_DIR=$home_dir/Project/results/imagenet_img_gradcam/split_7899_fold${FOLD}
# ─────────────────────────────────────────────────────────────────────────────

python $home_dir/Project/src/imagenet_img/train/gradcam.py \
    --head_checkpoint $HEAD_CHECKPOINT \
    --n_per_class     $N_PER_CLASS \
    --target          $TARGET \
    --out_dir         $OUT_DIR
