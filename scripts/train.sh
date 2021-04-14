export CUDA_VISIBLE_DEVICES=7
DATASET_PATH=data

SAVE_PATH=models/bert-base-chinese

MODEL_ARGS="--model_name_or_path bert-base-chinese \
            --tokenizer_name bert-base-chinese \
            --config_name bert-base-chinese \
            --max_seq_length 512"
DARA_ARGS
TRAIN_ARGS="--train \
            --eval \
            --predict \
            --per_device_train_batch_size 8 \
            --gradient_accumulation_steps 1 \
            --per_device_eval_batch_size 8 \
            --learning_rate 2e-5 \
            --num_train_epochs 3 \
            --lr_scheduler_type linear \
            --warmup 0.05 \
            --output_dir $SAVE_PATH \
            --save_steps 1000 \
            --tensorboard_dir $SAVE_PATH/tensorboard \
            --log_steps 10 \
            --eval_steps 500 \
            --seed 2020"

python main.py \
       $MODEL_ARGS \
       $TRAIN_ARGS