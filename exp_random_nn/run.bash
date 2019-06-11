#!/usr/bin/env bash

datasets_path="/scratch/ag919/datasets/"
out_path="./exp_random_nn/cifar10/"
config="cifar10"
n_samples=10000
seed=12

mkdir "$out_path"
for chans in 3 10 30 100; do
	echo "Running with ${chans} channels"
	python -m exp_random_nn.random_comparison --datasets_path="$datasets_path" \
		   --config="$config" --seed=$seed --channels=$chans --n_samples=$n_samples \
		   --out_path="$out_path"
done

python -m exp_random_nn.random_plot "$out_path/figure.pdf" \
	   "$out_path"/*_samples.csv "$out_path"/*_cov.csv
