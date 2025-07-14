#!/bin/bash

# Define the range of classes
# To improve efficiency, we distribute the generation of different classes across separate GPUs. You can change it to your own setting.
n=0
declare -a gpus=(0 1)
declare -a starts=($n $(($n+100)))
declare -a ends=($(($n+100)) $(($n+200)))

for i in ${!gpus[@]}; do
    gpu=${gpus[$i]}
    start=${starts[$i]}
    end=${ends[$i]}
    
    echo "Running on GPU $gpu with start=$start and end=$end"
    python generation/group_sampling.py --start $start --end $end --gpu $gpu &
done

wait
echo "All tasks completed."