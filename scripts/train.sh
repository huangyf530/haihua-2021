export CUDA_VISIBLE_DEVICES=7
DATASET_PATH=data

python main.py \
  --model_name_or_path voidful/albert_chinese_base \
  --train_file $DATASET_PATH/train.json \
  --validation_file $DATASET_PATH/validation.json \
  --max_seq_length 512 \
  --per_device_train_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir output \
  --seed 2020