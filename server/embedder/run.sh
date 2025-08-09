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
    elif [[ $user_input == "n" || $user_input == "N" ]]; then
        read -p "input a value for $guidance: " user_input
        export "$var_name=$user_input"
    else
        export "$var_name=$user_input"
    fi

    echo "Selected $guidance: ${var_name} = ${!var_name}"
    echo ""
}

set_variable_with_default "CUDA_VISIBLE_DEVICES" "CUDA_VISIBLE_DEVICES" "0" "1" "2" "3"

uvicorn embedder:app --host localhost --port 8000 --reload
