#!/bin/sh

clear

echo "##################################"
echo "#  AAA segmentation keep-alive  #"
echo "##################################"

echo "Activating python version 3.82 ..."
eval "$(pyenv init -)"
pyenv shell 3.8.2

export TF_CPP_MIN_LOG_LEVEL=1
export TF_KERAS=1

tmux neww -d 

COMMAND="python AAA_zone_predict.py"
EXIT_CODE=1
(while [ $EXIT_CODE -gt 0 ]; do
    now=$(date +"%c")
    echo "Start time : $now"
    echo "Executing '`$COMMAND`'"
    $COMMAND
    # loops on error code: greater-than 0
    EXIT_CODE=$?
    if [ $EXIT_CODE -gt 0 ]
    then
        now=$(date +"%c")
        echo -e "#####################################################\n\n\n\n\n\n\n\n\n\n\n\n"
        echo "$now: Kernel crashed"
        echo -e "\n\n\n\n\n\n\n\n\n\n\n\n#####################################################"
    else
        now=$(date +"%c")
        echo "$now: Script finished"
    fi
done)
