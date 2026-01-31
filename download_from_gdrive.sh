#!/bin/bash

# --- 1. Configuration ---
REMOTE="gdrive"
BASE_PATH="DF40"  # e.g. DF40

# Local save directory
SAVE_DIR="/tmp/f2256768/df40"
mkdir -p "$SAVE_DIR"

# Train/Test splits (zip list will be discovered automatically)
SPLITS=("DF40_train" "DF40")

# Other items directly under BASE_PATH (non-zip)
OTHER_ITEMS=(
  "Celeb-DF-v2"
  "FaceForensics++"
)

# --- 2. Start Pipeline Operations ---
cd "$SAVE_DIR" || exit 1
echo "Start downloading"

for SPLIT in "${SPLITS[@]}"; do
  # list all zip files under split
  mapfile -t ZIP_FILES < <(rclone lsf "${REMOTE}:${BASE_PATH}/${SPLIT}" --files-only | grep -E '\.zip$')
  for ZIP_NAME in "${ZIP_FILES[@]}"; do
    DIR_NAME="${ZIP_NAME%.*}"
    TARGET_DIR="${SPLIT}/${DIR_NAME}"
    TEMP_DIR="${SPLIT}/.tmp"

    echo "------------------------------------------------"

    # 1. Check if folder exists
    if [ -d "$TARGET_DIR" ]; then
      echo "Folder exists: $TARGET_DIR"
    else
      # 2. Download
      echo "Downloading: ${SPLIT}/${ZIP_NAME}"
      mkdir -p "$TEMP_DIR"
      rclone copy "${REMOTE}:${BASE_PATH}/${SPLIT}/${ZIP_NAME}" "$TEMP_DIR" --transfers=16 --stats 1m --stats-one-line

      # 3. Unzip to split folder
      if [ -f "${TEMP_DIR}/${ZIP_NAME}" ]; then
        echo "Unzipping to: $TARGET_DIR"
        mkdir -p "${SPLIT}"
        unzip -q "${TEMP_DIR}/${ZIP_NAME}" -d "$TARGET_DIR"

        # If the zip already contains a top-level folder with the same name, flatten it
        if [ -d "${TARGET_DIR}/${DIR_NAME}" ]; then
          echo "Flattening nested dir: ${TARGET_DIR}/${DIR_NAME}"
          shopt -s dotglob
          mv "${TARGET_DIR}/${DIR_NAME}"/* "$TARGET_DIR"/
          rmdir "${TARGET_DIR}/${DIR_NAME}"
          shopt -u dotglob
        fi

        echo "Deleting zip: $ZIP_NAME"
        rm "${TEMP_DIR}/${ZIP_NAME}"
      else
        echo "Download failed for ${SPLIT}/${ZIP_NAME}"
        continue
      fi
    fi
  done
done

# Directly copy non-zip items
for ITEM in "${OTHER_ITEMS[@]}"; do
  echo "------------------------------------------------"
  if [ -e "$ITEM" ]; then
    echo "Already exists: $ITEM"
  else
    echo "Downloading: $ITEM"
    rclone copy "${REMOTE}:${BASE_PATH}/${ITEM}" "$ITEM" --transfers=16 --stats 1m --stats-one-line
  fi
done

echo "------------------------------------------------"
echo "All downloads finished"
echo "Space Usage:"
du -sh .
