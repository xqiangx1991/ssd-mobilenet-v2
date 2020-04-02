#! /bin/bash

killall tensorboard

# rm -rf ./trained_model/*

nohup tensorboard --bind_all \
      --window_title=`hostname` \
      --logdir=./trained_model/ &
