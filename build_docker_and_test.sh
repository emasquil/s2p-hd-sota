# docker build
image=s2p-hd-nov2024
docker build -f ./Dockerfile -t $image .

# sample run
user=$(id -u)
docker run --gpus=all --user $user --rm --workdir /workdir -v $PWD:/workdir -ti $image \
       	s2p tests/data/input_pair/config.json


