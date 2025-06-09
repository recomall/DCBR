#!/bin/bash

datasets=("MealRec+H" "MealRec+L" "iFashion")
for dataset_name in "${datasets[@]}"; do
    python train.py -m DCBR -d $dataset_name
done