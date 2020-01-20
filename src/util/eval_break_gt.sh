#!/usr/bin/env bash

HL='\033[1;32m'  # highlight
NC='\033[0m' # No Color

modes="BASE PIR BAB"
for mode in $modes
do

    if [ "$mode" == "BAB" ]; then
        log_path_r="../../log/$mode/_1-r.txt"
        log_path_t="../../log/$mode/_1-t.txt"
    else
        log_path_r="../../log/$mode/_r.txt"
        log_path_t="../../log/$mode/_t.txt"
    fi

    echo -e "${HL}mode: $mode -- offset: r${NC}"
    stdbuf -oL python3 eval_break_gt.py $mode "r" > "$log_path_r"

    echo -e "${HL}mode: $mode -- offset: t${NC}"
    stdbuf -oL python3 eval_break_gt.py $mode "t" > "$log_path_t"
done