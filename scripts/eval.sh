#!/bin/bash
# export CUDA_VISIBLE_DEVICES=7
export TOKENIZERS_PARALLELISM=true
GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0

MODEL_NAME=hfl/chinese-bert-wwm-ext
DATASET_PATH=data
SEED=46
SAVE_PATH=models/longformer-seed$SEED
DATA_PATH=data

ENSEMBLE_MODELS=models/bert-base-chinese-seed45/best,models/bert-base-chinese-seed2020/best,models/bert-base-chinese/best,models/bert-base-chinese-seed46/best,models/bert-base-chinese-seed52/best

DISTRIBUTED_ARGS="-m torch.distributed.launch --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

MODEL_ARGS="--model_name_or_path $MODEL_NAME \
            --tokenizer_name $MODEL_NAME \
            --config_name $MODEL_NAME \
            --no_load_optim \
            --max_seq_length 512"

DATA_ARGS="--train_file $DATA_PATH/train.json \
           --predict_file $DATA_PATH/validation.json \
           --predict_out $DATA_PATH/prediction.csv \
           --cache_data_dir $DATA_PATH/cache \
           --pad_to_max_length"

EVAL_ARGS="--predict \
            --ensemble_models $ENSEMBLE_MODELS \
            --seed $SEED"

python \
       main.py \
       $DATA_ARGS \
       $MODEL_ARGS \
       $EVAL_ARGS