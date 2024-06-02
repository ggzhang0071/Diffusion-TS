img="nvcr.io/nvidia/pytorch:23.04-py3" 

docker run --gpus all  --privileged=true   --workdir /git --name "diffusion_ts"  -e DISPLAY --ipc=host -d --rm  -p 6333:8889\
 -v /home/ggzhang/Diffusion-TS:/git/diffusion_ts \
 -v /home/ggzhang/datasets:/git/datasets \
 $img sleep infinity

docker exec -it diffusion_ts   /bin/bash

#docker images  |grep "diffusion_ts"  |grep "21."

#docker stop  diffusion_ts

