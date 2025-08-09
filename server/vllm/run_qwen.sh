export HF_OFFLINE=1

set_variable_with_default() {
    local var_name=$1
    shift
    local guidance=$1
    shift
    local default_values=("$@")
    local num_options=${#default_values[@]}

    # 打印所有选项
    for ((i = 0; i < num_options; i++)); do
        echo "$((i + 1)). ${default_values[$i]}"
    done

    read -p "input $guidance (input 1-$num_options to choose a default value): " user_input

    if [[ $user_input =~ ^[1-$num_options]$ ]]; then
        local selected_index=$((user_input - 1))
        export "$var_name=${default_values[$selected_index]}"
    elif [[ $user_input == "n" ]]; then
        read -p "input $guidance: " user_input
        export "$var_name=$user_input"
    else
        export "$var_name=$user_input"
    fi

    echo "Selected $guidance: ${var_name} = ${!var_name}"
    echo ""
}

set_variable_with_default CUDA_VISIBLE_DEVICES device 0,1 2,3 4,5 6,7, 0,1,2,3 4,5,6,7 0,1,2,3,4,5,6,7
gpu_nums=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

tensor_parallel_size=$gpu_nums
echo "tensor_parallel_size: $tensor_parallel_size"

set_variable_with_default model model Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-72B-Instruct
echo "model: $model"

port=8003

export VLLM_ATTENTION_BACKEND=XFORMERS
vllm serve \
    $model \
    --served-model-name $model \
    --port $port \
    --tensor-parallel-size $tensor_parallel_size \
    --pipeline-parallel-size 1 \
    --gpu_memory_utilization 0.7 \
    --dtype auto
