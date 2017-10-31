#!/bin/bash

Room=("c133")

for i in "${Room[@]}"; do
    for j in {1..50}; do
        ping -c 1 -W 1 $i"-"$j > /dev/null 2>&1
        if [[ $? == 0 ]]
        then
            echo $i"-"$j
        fi
    done
done
