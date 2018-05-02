#!/bin/bash

# convert the images folder to the test.list and train.list file according to
#   the distribution, command will clear the train.list and test.list files first
#
#   Args:
#       path: the path to the video folder
#       percent: the percent of the data to be in the training set. 50 would be a half-half split.
#   Usage:
#       ./convert_images_to_list.sh path/to/video 4
#   Example Usage:
#       ./convert_images_to_list.sh ~/document/videofile 4
#   Example Output(train.list and test.list):
#       /Volumes/passport/datasets/action_kth/origin_images/boxing/person01_boxing_d1_uncomp 0
#       /Volumes/passport/datasets/action_kth/origin_images/boxing/person01_boxing_d2_uncomp 0
#       ...
#       /Volumes/passport/datasets/action_kth/origin_images/handclapping/person01_handclapping_d1_uncomp 1
#       /Volumes/passport/datasets/action_kth/origin_images/handclapping/person01_handclapping_d2_uncomp 1
#       ...

> train_$2.list
> test_$2.list
COUNT=-1
for folder in $1/*
do
    COUNT=$[$COUNT + 1]

    echo $folder
    for imagesFolder in "$folder"/*
    do
        if (( $(shuf -i 1-100 -n 1) > $2 )); then
            echo "$imagesFolder" $COUNT >> test_$2.list
        else
            echo "$imagesFolder" $COUNT >> train_$2.list
        fi        
    done
done
