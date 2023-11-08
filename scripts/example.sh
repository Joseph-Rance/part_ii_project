#!/bin/bash
# ray start --head
bash scripts/get_adult.sh > outputs/download
echo "running example experiment"
python src/main.py configs/example.yaml -c $1 -g $2