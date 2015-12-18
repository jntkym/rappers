#!/bin/sh
NICE="nice -n 19"

temp_feature_dir=/data/$USER/rapper/features
feature_dir=/zinnia/$USER/rapper/features
all_features_file=/zinnia/$USER/rapper/all_features.dat
feature_extract_script=./feature_extract.py
task_file=./feature_extract.task

# make directory. 
mkdir -p $feature_dir
gxpc e "mkdir -p $temp_feature_dir"

# generate task file, print.
$NICE python -c 'import feature_extract; feature_extract.generate_task()' > $task_file
gxpc js -a work_file=feature_extract.task -a cpu_factor=0.125

# aggregate the result.
gxpc e "cp $temp_feature_dir/* $feature_dir"
for file in $(ls $feature_dir/*)
do
    cat $file >> $all_features_file
done

#train model
model=/zinnia/rapper/model_NN3
svm_learn_light -z p $all_features_file $model 

#delete files in temp directory.
gxpc e "rm -rf $temp_feature_dir"

