#!/bin/bash
cd /home/aryana/Documents/GitHub/roman_grism_sr
source /home/aryana/Documents/GitHub/super_resolution/sup_res/bin/activate

echo "=== SR1 fine-tuning on Roman inputs ===" | tee train/roman_sr1/finetune.log
python scripts/finetune_sr1_roman.py \
    --epochs 50 \
    --wandb_mode online \
    --wandb_project roman_grism_sr \
    --wandb_name sr1_roman_finetune 2>&1 | tee -a train/roman_sr1/finetune.log

echo "=== Z-head v3 (on fine-tuned SR1) ===" | tee -a train/roman_sr1/finetune.log
python scripts/train_roman_zhead.py \
    --sr1_ckpt train/roman_sr1/best_sr1_roman.pth \
    --wandb_mode online \
    --wandb_project roman_grism_sr \
    --wandb_name roman_zhead_v3 2>&1 | tee -a train/roman_sr1/finetune.log
