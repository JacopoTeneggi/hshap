#!/bin/sh

for EXPL_ID in 2 3 4
do
    for REF_SIZE in 100 200 400 800 1600
    do
        python /home/jacopo/repo/hshap/hshap/experiments/rsna/LOR/compute_LOR_drop_single_explainer.py "/home/jacopo" 1 $REF_SIZE 20 $EXPL_ID
    done
done