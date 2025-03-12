#!/bin/bash
# Richiedere una GPU e avviare Jupyter Notebook
srun -Q --immediate=10 --partition=all_serial --gres=gpu:2 --account=ai4bio2024 --time=120:00 --pty bash -c '
  source /homes/fmorandi/.bashrc &&
  jupyter notebook --ip=0.0.0.0 --no-browser --port=8888
'