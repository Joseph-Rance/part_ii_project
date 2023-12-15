if [ ! -d "/datasets/FedScale/reddit/reddit" ]; then
    wget -O /datasets/FedScale/reddit/reddit.tar.gz https://fedscale.eecs.umich.edu/dataset/reddit.tar.gz
    tar -xf /datasets/FedScale/reddit/reddit.tar.gz -C /datasets/FedScale/reddit
    rm -f /datasets/FedScale/reddit/reddit.tar.gz
fi