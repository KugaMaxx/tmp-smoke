python3 train-clip.py \
    --output_dir ./clip-time-series-custom-tokenizer \
    --dataset_name "/home/dszh/workspace/tmp-smoke/Python/data/cube-texture" \
    --do_train --do_eval \
    --per_device_train_batch_size="64" \
    --per_device_eval_batch_size="64" \
    --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
    --overwrite_output_dir \
    --remove_unused_columns=False \
    --num_train_epochs 30
