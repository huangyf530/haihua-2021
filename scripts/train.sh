#!/bin/bash
# export CUDA_VISIBLE_DEVICES=7
export TOKENIZERS_PARALLELISM=true
GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0

DATASET_PATH=data

SAVE_PATH=models/bert-base-chinese
DATA_PATH=data

DISTRIBUTED_ARGS="-m torch.distributed.launch --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

MODEL_ARGS="--model_name_or_path hfl/chinese-bert-wwm-ext \
            --tokenizer_name hfl/chinese-bert-wwm-ext \
            --config_name hfl/chinese-bert-wwm-ext \
            --max_seq_length 512"

DATA_ARGS="--train_file $DATA_PATH/train.json \
           --predict_file $DATA_PATH/validation.json \
           --cache_data_dir $DATA_PATH/cache \
           --pad_to_max_length"

TRAIN_ARGS="--train \
            --eval \
            --predict \
            --per_device_train_batch_size 8 \
            --gradient_accumulation_steps 4 \
            --per_device_eval_batch_size 16 \
            --weight_decay 0.0001 \
            --learning_rate 2e-5 \
            --num_train_epochs 10 \
            --lr_scheduler_type linear \
            --warmup 0.05 \
            --output_dir $SAVE_PATH \
            --save_steps 1000 \
            --tensorboard_dir $SAVE_PATH/tensorboard \
            --log_steps 10 \
            --eval_steps 100 \
            --seed 2020"

python \
       main.py \
       $DATA_ARGS \
       $MODEL_ARGS \
       $TRAIN_ARGS