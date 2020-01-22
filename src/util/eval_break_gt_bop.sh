#!/usr/bin/env bash

HL='\033[1;32m'  # highlight
NC='\033[0m' # No Color

modes="BAB_patch0.2"  #"BASE PIR BAB"
for mode in $modes
do
    for (( offset=5; offset<=5; offset=offset+5 ))
    do
        if [ "$mode" == "BAB_patch0.2" ]; then
            result_path_r=$(printf "/home/dominik/projects/hsr-grasping/log/$mode/1-r%0.2d_lm-test.csv" $offset)
            result_path_t=$(printf "/home/dominik/projects/hsr-grasping/log/$mode/1-t%0.2d_lm-test.csv" $offset)

            log_path_r=$(printf "/home/dominik/projects/hsr-grasping/log/$mode/_1-r_%0.2d_bop.txt" $offset)
            log_path_t=$(printf "/home/dominik/projects/hsr-grasping/log/$mode/_1-t_%0.2d_bop.txt" $offset)
        else
            result_path_r=$(printf "/home/dominik/projects/hsr-grasping/log/$mode/r%0.2d_lm-test.csv" $offset)
            result_path_t=$(printf "/home/dominik/projects/hsr-grasping/log/$mode/t%0.2d_lm-test.csv" $offset)

            log_path_r=$(printf "/home/dominik/projects/hsr-grasping/log/$mode/_r_%0.2d_bop.txt" $offset)
            log_path_t=$(printf "/home/dominik/projects/hsr-grasping/log/$mode/_t_%0.2d_bop.txt" $offset)
        fi

        echo -e "${HL}mode: $mode -- offset: r $offset${NC}"
        echo $result_path_r
        echo $log_path_r
        stdbuf -oL python ./scripts/eval_bop19.py --result_filenames=$result_path_r > "$log_path_r"
        ap=$(tail -n 6 "$log_path_r" | grep -o "bop19_average_recall:[^;]*" | grep -o "0.[^;]*" | bc -l)
        bc -l <<< "$ap*100.0"
        mvsd=$(tail -n 6 "$log_path_r" | grep -o "bop19_average_recall_vsd:[^;]*" | grep -o "0.[^;]*" | bc -l)
        bc -l <<< "$mvsd*100.0"
        mspd=$(tail -n 6 "$log_path_r" | grep -o "bop19_average_recall_mspd:[^;]*" | grep -o "0.[^;]*" | bc -l)
        bc -l <<< "$mspd*100.0"
        mssd=$(tail -n 6 "$log_path_r" | grep -o "bop19_average_recall_mssd:[^;]*" | grep -o "0.[^;]*" | bc -l)
        bc -l <<< "$mssd*100.0"

        echo -e "${HL}mode: $mode -- offset: t${NC}"
        echo $result_path_t
        echo $log_path_t
        stdbuf -oL python ./scripts/eval_bop19.py --result_filenames=$result_path_t > "$log_path_t"
        ap=$(tail -n 6 "$log_path_t" | grep -o "bop19_average_recall:[^;]*" | grep -o "0.[^;]*" | bc -l)
        bc -l <<< "$ap*100.0"
        mvsd=$(tail -n 6 "$log_path_t" | grep -o "bop19_average_recall_vsd:[^;]*" | grep -o "0.[^;]*" | bc -l)
        bc -l <<< "$mvsd*100.0"
        mspd=$(tail -n 6 "$log_path_t" | grep -o "bop19_average_recall_mspd:[^;]*" | grep -o "0.[^;]*" | bc -l)
        bc -l <<< "$mspd*100.0"
        mssd=$(tail -n 6 "$log_path_t" | grep -o "bop19_average_recall_mssd:[^;]*" | grep -o "0.[^;]*" | bc -l)
        bc -l <<< "$mssd*100.0"
    done
done

# example: get only results -> $ tail -n 6 projects/hsr-grasping/log/PIR/_r_00_bop.txt | grep -o " 0.[^;]*"