#!/bin/bash

sudo killall rosmaster
sudo killall gzserver
sudo killall gzclient
roslaunch room_evac_gazebo obstacle_avoidance.launch
