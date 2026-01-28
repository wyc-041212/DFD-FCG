#!/bin/bash
#SBATCH --job-name=dfd-fcg_df40
#SBATCH --output=dfd-fcg_df40_%j.out
#SBATCH --error=dfd-fcg_df40_%j.err
#SBATCH --partition=short
#SBATCH --nodelist=gpu20         # Force execution on node gpu20
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00

# --- Configuration ---
TARGET_TMP_DIR="/tmp/f2256768"
DATA_ROOT="${TARGET_TMP_DIR}/df40"
DOWNLOAD_SCRIPT="/home/comp/f2256768/DFD-FCG/download_from_gdrive.sh"
JSON_ROOT="/home/comp/f2256768/DF40/DeepfakeBench_DF40/preprocessing/dataset_json"

# Protocol settings
TRAIN_PROTOCOL="FSAll_ff"
TEST_PROTOCOL="FSAll_ff"
PROTOCOL_COMPRESSION="c23"

echo "Job Start on Node: $(hostname)"
source /usr/local/anaconda/3-2024.06/bin/activate dfd-fcg

# --- 1. Smart Check: Is data persisted? ---
if [ -d "$DATA_ROOT" ] && [ "$(ls -A $DATA_ROOT)" ]; then
    echo "Directory $DATA_ROOT detected and not empty."
    echo "Skipping download, utilizing existing data."
else
    echo "Directory $DATA_ROOT missing or empty (likely cleaned by system)."
    echo "Calling download script: download_from_gdrive.sh ..."
    
    # Grant execution permission and run download script
    chmod +x "$DOWNLOAD_SCRIPT"
    bash "$DOWNLOAD_SCRIPT"
    
    # Secondary Check
    if [ ! -d "$DATA_ROOT" ]; then
        echo "Critical Error: Directory still missing after download script execution!"
        exit 1
    fi
fi

# --- 2. Run training ---
echo "------------------------------------------------"
echo "Starting DFD-FCG DF40 training..."

cd /home/comp/f2256768/DFD-FCG || exit 1

python -m main \
  --config configs/base.yaml \
  --config configs/clip/L14/ffg.yaml \
  --config configs/data_df40.yaml \
  --trainer.devices=1 \
  --data.init_args.train_datamodules.0.init_args.data_dir="$DATA_ROOT" \
  --data.init_args.val_datamodules.0.init_args.data_dir="$DATA_ROOT" \
  --data.init_args.test_datamodules.0.init_args.data_dir="$DATA_ROOT" \
  --data.init_args.train_datamodules.0.init_args.protocol_json_folder="$JSON_ROOT" \
  --data.init_args.val_datamodules.0.init_args.protocol_json_folder="$JSON_ROOT" \
  --data.init_args.test_datamodules.0.init_args.protocol_json_folder="$JSON_ROOT" \
  --data.init_args.train_datamodules.0.init_args.protocol_dataset_name="$TRAIN_PROTOCOL" \
  --data.init_args.val_datamodules.0.init_args.protocol_dataset_name="$TEST_PROTOCOL" \
  --data.init_args.test_datamodules.0.init_args.protocol_dataset_name="$TEST_PROTOCOL" \
  --data.init_args.train_datamodules.0.init_args.protocol_compression="$PROTOCOL_COMPRESSION" \
  --data.init_args.val_datamodules.0.init_args.protocol_compression="$PROTOCOL_COMPRESSION" \
  --data.init_args.test_datamodules.0.init_args.protocol_compression="$PROTOCOL_COMPRESSION"

echo "Job Finished"
