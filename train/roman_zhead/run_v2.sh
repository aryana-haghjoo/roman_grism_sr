#!/bin/bash
cd /home/aryana/Documents/GitHub/roman_grism_sr

echo "=== Generating roman_train_spectra.npz (using .venv with romanisim) ===" | tee train/roman_zhead/train_v2.log
/home/aryana/Documents/GitHub/roman_grism_sr/.venv/bin/python \
    scripts/make_roman_spectra.py --split train 2>&1 | tee -a train/roman_zhead/train_v2.log

echo "=== Starting z-head training v2 (using sup_res venv) ===" | tee -a train/roman_zhead/train_v2.log
source /home/aryana/Documents/GitHub/super_resolution/sup_res/bin/activate
python scripts/train_roman_zhead.py \
    --wandb_mode online \
    --wandb_project roman_grism_sr \
    --wandb_name roman_zhead_v2 2>&1 | tee -a train/roman_zhead/train_v2.log
