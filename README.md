# hsr-grasping

## Build Dockerfile with global context
In hsr-grasping:

`docker build -f src/densefusion/Dockerfile . -t densefusion`

## Run densefusion
`docker run -it --net=host --volume=/home/dominik/projects/hsr-grasping:/hsr-grasping densefusion /bin/bash`

## Run verefine
`xhost +local:'hostname'` \
`docker run -it --net=host --env='DISPLAY' --env='QT_X11_NO_MITSHM=1' --volume=/home/dominik/projects/hsr-grasping:/hsr-grasping verefine /bin/bash`