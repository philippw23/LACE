#!/bin/bash
#SBATCH --job-name=imagenet_img
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --exclude=aioserver2
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# ── Parameters (edit here) ────────────────────────────────────────────────────
SPLITS="/mnt/nfs/homedirs/philippw/Project/data/internal_dataset/split_binary.json"
BTXRD_MANIFEST="/mnt/nfs/homedirs/philippw/Project/data/BTXRD/btxrd_downstream_binary.json"
BINARY=true

EPOCHS=150
PATIENCE=30
BATCH_SIZE=32
LR_MLP=0.0001646564531459077
WEIGHT_DECAY=0.0835001362514313
DROPOUT=0.3
HIDDEN_DIMS="[64, 32]"
META_EMBED_DIM=0 # only used if head is mlp, e.g 16

# Head mode: mlp (age+sex fusion), mlp_no_meta (image only)
HEAD="mlp_no_meta"
# Loss: ce, wce, focal, cb_focal, balanced_softmax, ldam
LOSS="focal"
# Class weighting: none, inverse, sqrt, effective
CLASS_WEIGHTING="inverse"
FOCAL_GAMMA=3
SEED=42
EARLY_STOPPING_METRIC="val_bal_acc"
# ─────────────────────────────────────────────────────────────────────────────

home_dir="/mnt/nfs/homedirs/$USER"
export HOME=$home_dir
cd ${SLURM_SUBMIT_DIR}

LOG_FILE="$home_dir/Project/logs/slurm-${SLURM_JOBID}_imagenet_img_${HEAD}_${LOSS}.out"
exec > "$LOG_FILE" 2>&1

echo "Starting job ${SLURM_JOBID} — encoder: ViT-B/16 (ImageNet pretrained) | head: $HEAD"
squeue -j ${SLURM_JOBID} -O nodelist | tail -n +2

MY_CONDA_ENV="master"
export SSL_CERT_FILE=$home_dir/miniconda3/envs/$MY_CONDA_ENV/ssl/cert.pem
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export HF_HOME=$home_dir/.cache/huggingface
export TRANSFORMERS_CACHE=$home_dir/.cache/huggingface/transformers
export WANDB_DIR=$home_dir/Project/logs
export PATH=$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin:$home_dir/miniconda3/bin:$PATH
export PYTHONPATH=$home_dir/Project/src
unset PYTORCH_NVML_BASED_CUDA_CHECK
echo "Environment: $MY_CONDA_ENV"

$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin/python $home_dir/Project/src/imagenet_img/train/downstream.py \
    --head           $HEAD \
    --splits         $SPLITS \
    --out_dir        $home_dir/Project/results/imagenet_img \
    --epochs         $EPOCHS \
    --patience       $PATIENCE \
    --batch_size     $BATCH_SIZE \
    --lr_mlp         $LR_MLP \
    --weight_decay   $WEIGHT_DECAY \
    --dropout        $DROPOUT \
    --hidden_dims    $HIDDEN_DIMS \
    --meta_embed_dim $META_EMBED_DIM \
    --loss           $LOSS \
    --class_weighting $CLASS_WEIGHTING \
    --focal_gamma    $FOCAL_GAMMA \
    --early_stopping_metric $EARLY_STOPPING_METRIC \
    --binary         $BINARY \
    --btxrd_manifest $BTXRD_MANIFEST \
    --eval_test \
    --use_mask \
    --seed           $SEED \
    --wandb \
    --wandb_project  imagenet-img-downstream \
    --wandb_entity   philipp-wiese
