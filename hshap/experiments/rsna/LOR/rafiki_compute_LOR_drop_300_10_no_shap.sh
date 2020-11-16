#!/bin/sh

for REF_SIZE in 100 200 400 800 1600
    do
        python /home/jacopo/repo/hshap/hshap/experiments/rsna/LOR/compute_LOR_drop_no_shap.py "/home/jacopo" 300 $REF_SIZE 10
    done
done