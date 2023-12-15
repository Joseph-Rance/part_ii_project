#!/bin/bash
# ray start --head
echo "running example experiment"
python src/main.py configs/example.yaml -c $1 -g $2