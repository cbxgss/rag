DEFAULT_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
read -p "input CUDA_VISIBLE_DEVICES (input y to use default: $DEFAULT_CUDA_VISIBLE_DEVICES): " CUDA_VISIBLE_DEVICES
if [ "$CUDA_VISIBLE_DEVICES" = "y" ]; then
    export CUDA_VISIBLE_DEVICES=$DEFAULT_CUDA_VISIBLE_DEVICES
else
    export CUDA_VISIBLE_DEVICES
fi

num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c)
num_devices=$((num_devices + 1))
tensor_parallel_size=$num_devices
echo "tensor_parallel_size: $tensor_parallel_size"

DEFAULT_port=8003
read -p "input port (input y to use default: $DEFAULT_port): " port
if [ "$port" = "y" ]; then
    export port=$DEFAULT_port
else
    export port
fi

    # --limit_mm_per_prompt image=4 \
    # --max_model_len 32768 \
vllm serve \
    mistralai/Mistral-Small-24B-Instruct-2501 \
    --served-model-name mistral-small \
    --port $port \
    --enforce-eager \
    --tensor-parallel-size $tensor_parallel_size \
    --pipeline-parallel-size 1 \
    --gpu_memory_utilization 0.7 \
    --dtype auto
