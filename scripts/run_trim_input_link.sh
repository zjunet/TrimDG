#!/bin/bash

# 确保日志目录存在
mkdir -p result

gpu=$1
dataset_name=$2

# 定义要遍历的参数列表
model_names=("TGAT" )
dataset_names=($2)  # 可以添加更多数据集名称
num_neighbors_list=(20)  # 邻居数量参数
target_ratios=(1.0)  # 目标比例参数
gpus=($1)  # GPU编号，如有多个GPU可添加更多

for model in "${model_names[@]}"; do
    for dataset in "${dataset_names[@]}"; do
        for neighbors in "${num_neighbors_list[@]}"; do
            for ratio in "${target_ratios[@]}"; do
                for gpu in "${gpus[@]}"; do
                    # 构建日志文件名
                    log_file="log/our_link_${model}_${dataset}_knn${neighbors}_target${ratio}.out"
                    cmd="python -u trim_link.py \
                        --model_name $model \
                        --dataset_name $dataset \
                        --num_runs 3 \
                        --cache 1 \
                        --sample_neighbor_strategy recent \
                        --num_neighbors $neighbors \
                        --gpu $gpu \
                        --num_epochs 50 \
                        --presampling_total_rate 0.6 \
                        --batch_size 200 \
                        --batch_rate 0.5 \
                        --batch_sampling 1 \
                        > $log_file"

                    echo "执行命令: $cmd"
                    eval $cmd

                done
            done
        done
    done
done

echo "所有实验已启动"
