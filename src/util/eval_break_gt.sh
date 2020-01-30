#!/usr/bin/env bash

HL='\033[1;32m'  # highlight
NC='\033[0m' # No Color

scenes=(1 2 4 5 6 8 9 10 11 12 13 14 15)
modes="BAB"  #"BEST ALLON EVEN BAB"  #"PIR BAB MITASH"  # "BASE PIR BAB"
for mode in $modes
do

    if [ "$mode" == "BAB" ]; then
        log_path_r="../../log/$mode/_1-r.txt"
        log_path_t="../../log/$mode/_1-t.txt"


    else
        log_path_r="../../log/$mode/_r.txt"
        log_path_t="../../log/$mode/_t.txt"
    fi

    for scene in ${scenes[@]}; do
        echo -e "${HL}mode: $mode -- offset: r -- scene: $scene${NC}"
        stdbuf -oL python3 eval_break_gt.py $mode "r" $scene > "$log_path_r"

        echo -e "${HL}mode: $mode -- offset: t -- scene: $scene${NC}"
        stdbuf -oL python3 eval_break_gt.py $mode "t" $scene > "$log_path_t"
    done
done