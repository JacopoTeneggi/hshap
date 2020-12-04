#!/bin/sh

for REF_SIZE in 100 200 400 800 1600
do
	python /cis/home/jteneggi/repo/hshap/hshap/experiments/rsna/LOR/compute_LOR_drop_single_explainer.py "gaon" 300 $REF_SIZE 20 1
done
