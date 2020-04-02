#! /bin/bash

killall tensorboard

# rm -rf ./trained_model/*

nohup /home/xuchen.xq/python_venv/tf2.x/bin/tensorboard --bind_all \
      --window_title=`hostname` \
      --logdir=./trained_model/ &
