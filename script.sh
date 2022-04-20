#!/bin/bash

# read -p 'Specfiy the input video path: ' inputPathVar
# read -p 'Specfiy the output video path: ' outputPathVar
# read -p 'Use debugging mode [y/n]: ' debugVar
echo -e "installing the dependants packges !! \n"

pip install opencv-python
pip install moviepy

echo -e "\n Starting execution \n "

python python_script.py $1 $2 $3 > /dev/null

echo -e "\n execution finished \n "


