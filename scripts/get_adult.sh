#!/bin/bash
# this script downloads the adult census dataset
[ -d "data/adult" ] || (echo "added directory 'data/adult'" && (mkdir data; mkdir data/adult))
wget https://archive.ics.uci.edu/static/public/2/adult.zip -O data/adult.zip -q
unzip -o data/adult.zip -d data/adult
sed '1d' data/adult/adult.test > data/adult/adult.temp
mv data/adult/adult.temp data/adult/adult.test