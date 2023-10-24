sudo apt-get install unzip
wget https://archive.ics.uci.edu/static/public/2/adult.zip -O data/adult.zip
unzip data/adult.zip
sed '1d' data/adult/adult.test