#!/bin/bash
mkdir data/adult
wget https://archive.ics.uci.edu/static/public/2/adult.zip -O data/adult.zip
unzip -o data/adult.zip -d data/adult
sed '1d' data/adult/adult.test > data/adult/adult.test