#!/usr/bin/env bash

HL='\033[1;32m'  # highlight
NC='\033[0m' # No Color

modes="BAB_top0.3" #"BEST ALLON EVEN BAB"  #"BASE PIR BAB"
n_hyp=1 #5
offsets=(5) #(0 10 15 20 30 35 40 45)
for mode in $modes
do
    for offset in ${offsets[@]} #(( offset=50; offset<=50; offset=offset+5 ))
    do
        if [ "$mode" == "BAB_top0.3" ]; then
            result_path_r=$(printf "/home/dominik/projects/hsr-grasping/log/$mode/%i-r%0.2d_lm-test.csv" $n_hyp $offset)
            result_path_t=$(printf "/home/dominik/projects/hsr-grasping/log/$mode/%i-t%0.2d_lm-test.csv" $n_hyp $offset)

            log_path_r=$(printf "/home/dominik/projects/hsr-grasping/log/$mode/_%i-r_%0.2d_bop.txt" $n_hyp $offset)
            log_path_t=$(printf "/home/dominik/projects/hsr-grasping/log/$mode/_%i-t_%0.2d_bop.txt" $n_hyp $offset)
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