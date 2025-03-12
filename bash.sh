#!/bin/bash

srun -Q --immediate=10 --partition=all_serial --gres=gpu:1 --account=ai4bio2024 --time 120:00 --pty bash
