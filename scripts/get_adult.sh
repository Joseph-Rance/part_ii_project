mkdir data/adult
wget https://archive.ics.uci.edu/static/public/2/adult.zip -O data/adult.zip
unzip data/adult.zip -o data/adult
sed '1d' data/adult/adult.test