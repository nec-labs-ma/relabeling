#!/bin/bash

# Directory for logs
OUTPUT_DIR=logs
mkdir -p "$OUTPUT_DIR"

# Model list
models=(
  "OpenGVLab/InternVL3-9B"
)

# models=(
#   "OpenGVLab/InternVL3-1B"
#   "OpenGVLab/InternVL3-2B"
#   "OpenGVLab/InternVL3-8B"
#   "OpenGVLab/InternVL3-9B"
#   "OpenGVLab/InternVL3-14B"
# )

# models=(
#   "google/gemma-3-4b-it"
#   "google/gemma-3-12b-it"
# )


# Validation Folder
OUTPUT_DIR="/net/acadia14a/data/sparsh/Relabeling/tests"
mkdir -p $OUTPUT_DIR


for model in "${models[@]}"; do
  # Clean model name for use in file names
  echo "model: $model"
  model_id=$(echo "$model" | tr '/' '-' )
  sbatch --job-name=bdd_9b --cpus-per-task=4 --mem=100GB -t 24-00:00:00 --partition=gpu --gres=gpu:2 --constraint=A6000 --exclude=ma-gpu27,ma-gpu17 --output="$OUTPUT_DIR/${model_id}_bdd.out" --error="$OUTPUT_DIR/${model_id}_bdd.err"  --wrap="python Relabeling_internvl-bdd.py ${model}"
done


# # Get all subdirectories in IMG_DIR starting with 'Run_'
# ALL_SUBDIRS=('OpenGVLab/InternVL3-1B' 'OpenGVLab/InternVL3-2B' 'OpenGVLab/InternVL3-8B' 'OpenGVLab/InternVL3-9B' 'OpenGVLab/InternVL3-14B')

# # Convert the space-separated string to a Bash array
# IFS=' ' read -r -a ALL_SUBDIRS_ARRAY <<< "$ALL_SUBDIRS"

# # Calculate the number of batches
# NUM_BATCHES=$(((${#ALL_SUBDIRS_ARRAY[@]} + BATCH_SIZE - 1) / BATCH_SIZE))

# # Print the total number of subdirectories and batches
# echo "Total subdirectories found: ${#ALL_SUBDIRS_ARRAY[@]}"
# echo "Total number of batches: $NUM_BATCHES"

# #Submit a job for each batch
# for ((i=0; i<NUM_BATCHES; i++))
# do
#   # Create a batch array slice
#   BATCH_SUBDIRS=("${ALL_SUBDIRS_ARRAY[@]:i*BATCH_SIZE:BATCH_SIZE}")
  
#   # Convert batch subdirectories to space-separated string
#   BATCH_SUBDIRS_STR=$(IFS=" "; echo "${BATCH_SUBDIRS[*]}")
#   echo "BATCH_SUBDIRS_STR: $BATCH_SUBDIRS_STR"
#   # # Print the batch being processed
#   # echo "Processing batch $i with subdirectories:"
#   # echo "${BATCH_SUBDIRS_STR}"

#   # Submit the job with sbatch
#   #sbatch --job-name=relabeling --cpus-per-task=1 --mem=50GB --partition=gpu --gres=gpu:2 --output=$OUTPUT_DIR/batch_$i.out --error=$OUTPUT_DIR/batch_$i.err --wrap="python Relabeling_internvl.py --subdirs $BATCH_SUBDIRS_STR"
# done
