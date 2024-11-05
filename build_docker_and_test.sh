# docker build
image=s2p-hd-nov2024
docker build -f ./Dockerfile -t $image .

# sample run
# output should be at:     tests/tests/testoutput/output_pair
user=$(id -u)
docker run --gpus=all --user $user --rm --workdir /workdir -v $PWD/tests:/workdir -ti $image \
       	s2p data/input_pair/config.gpu.json



# convert s2p-output-to-potree
# potree output should be at:     tests/tests/testoutput/output_pair/cloud.potree
docker run --gpus=all --user $user --rm --workdir /workdir -v $PWD/tests:/workdir -ti $image \
	/home/s2p-hd/utils/s2p_to_potree.py  --outdir testoutput/output_pair/ testoutput/output_pair/

