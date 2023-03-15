#!/bin/bash

xhost local:docker
cd $(rospack find locateobject) && docker-compose run locate_object
