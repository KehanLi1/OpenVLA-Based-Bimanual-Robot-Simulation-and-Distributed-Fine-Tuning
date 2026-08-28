#!/bin/bash

base_checkpoint=${1:-openvla/openvla-7b}
lora_finetuned_checkpoint_dir=${2:-ckpt}
save_path=${3:-$lora_finetuned_checkpoint_dir}

python vla-scripts/merge_lora_weights_and_save.py \
  --base_checkpoint "$base_checkpoint" \
  --lora_finetuned_checkpoint_dir "$lora_finetuned_checkpoint_dir" \
  --save_path "$save_path"
