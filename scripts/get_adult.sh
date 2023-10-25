#!/bin/bash
[ -d "data/adult" ] || (echo "added directory 'data/adult'" && mkdir data/adult)
wget https://archive.ics.uci.edu/static/public/2/adult.zip -O data/adult.zip > outputs/download  # TODO!!: does this actually solve the output issue?
unzip -o data/adult.zip -d data/adult
sed '1d' data/adult/adult.test > data/adult/adult.test