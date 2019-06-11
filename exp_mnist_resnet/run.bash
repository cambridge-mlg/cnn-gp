#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES="0,1"
datasets_path="/scratch/ag919/datasets/"
out_path="/scratch/ag919/grams_pytorch/mnist_test3"
config="mnist"
batch_size=200

if [ -d "$out_path" ]; then
	echo "Careful: directory \"$out_path\" already exists"
	exit 1
fi

space_separated_cuda="${CUDA_VISIBLE_DEVICES//,/ }"
n_workers=$(echo $space_separated_cuda | wc -w)
if [ "$n_workers" == 0 ]; then
	echo "You must specify CUDA_VISIBLE_DEVICES"
	exit 1
fi

echo "Downloading dataset"
python -c "import configs.$config as c; import cnn_gp; cnn_gp.DatasetFromConfig(\"$datasets_path\", c)"

echo "Starting kernel computation workers in parallel"

mkdir "$out_path"
worker_rank=0
for cuda_i in $space_separated_cuda; do
	this_worker="${out_path}/$(printf "%02d_nw%02d.h5" $worker_rank $n_workers)"

	CUDA_VISIBLE_DEVICES=$cuda_i python -m exp_mnist_resnet.save_kernel --n_workers=$n_workers \
		 --worker_rank=$worker_rank --datasets_path="$datasets_path" --batch_size=$batch_size \
		 --config="$config" --out_path="$this_worker" &
	pids[${i}]=$!
	worker_rank=$((worker_rank+1))
done
# Wait for all workers
for pid in ${pids[*]}; do
	wait $pid
done

echo "combining all data sets in one"
python -m exp_mnist_resnet.merge_h5_files "${out_path}"/*


echo "Classify using the complete set"
combined_file="${out_path}/$(printf "%02d_nw%02d.h5" 0 $n_workers)"
python -m exp_mnist_resnet.classify_gp --datasets_path="$datasets_path" \
	   --config="$config" --in_path="$combined_file"
