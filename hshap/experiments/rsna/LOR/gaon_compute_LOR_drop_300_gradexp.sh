#!/bin/sh

for EXPL_ID in 0
do
    for REF_SIZE in 100 200 400 800 1600
    do
        python /cis/home/jteneggi/repo/hshap/hshap/experiments/rsna/LOR/compute_LOR_drop_single_explainer.py "gaon" 300 $REF_SIZE 20 $EXPL_ID 5
    done
done