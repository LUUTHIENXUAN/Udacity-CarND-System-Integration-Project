#!/bin/bash

cd CarND_Capstone/ros
#catkin_make
source devel/setup.sh
roslaunch launch/styx.launch &

#open rviz
#gnome-terminal && disown && rviz
