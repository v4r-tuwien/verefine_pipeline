XAUTH=/tmp/.docker.xauth
if [ ! -f $XAUTH ]
then
    xauth_list=$(xauth nlist :0 | sed -e 's/^..../ffff/')
    if [ ! -z "$xauth_list" ]
    then
        echo $xauth_list | xauth -f $XAUTH nmerge -
    else
        touch $XAUTH
    fi
    chmod a+r $XAUTH
fi

docker run --gpus all \
           --network host \
           --privileged \
           -e QT_X11_NO_MITSHM=1 \
           -e DISPLAY=$DISPLAY \
           -e XAUTHORITY=$XAUTH \
           -v $XAUTH:$XAUTH \
           -v /tmp/.X11-unix/:/tmp/.X11-unix \
           -v $HOME/.Xauthority:/root/.Xauthority \
           -it verefine /bin/bash
