#!/bin/bash

save_dir = '/scratch/zhao.lin1/ddim_inversion_statistic'
pretrained_path = '/scratch/zhao.lin1/DiT-XL-2-256'

# the range of class ids
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
    python generation/dit_inversion_save_statistic.py --start $start --end $end --gpu $gpu --save_dir $save_dir --pretrained_path $pretrained_path &
done

# waiting for all tasks
wait
echo "All tasks completed."