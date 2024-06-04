#!/bin/bash

# Specify the log directory and port
LOG_DIR="/home/hiddenleaf/ARYAN_MT22019/hifi-gan/cp_hifigan/logs"
PORT=6006

LOG_DIRQ="/home/hiddenleaf/ARYAN_MT22019/hifi-ganQ/cp_hifigan7/logs"
PORTQ=6007



# Run tensorboard command
tensorboard --logdir="$LOG_DIR" --port="$PORT" &

tensorboard --logdir="$LOG_DIRQ" --port="$PORTQ" & 


