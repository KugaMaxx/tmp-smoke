# Run a Benchmark

This is a replica of how to reproduce benchmark results reported in the paper.

## Select a Model to Train

```bash
python3 train.py \
    --model_name <model_name> \
    --config_path <config_path> \
    --dataset_name_or_path <dataset_name> \
    --output_dir <output_dir> \
    --train_batch_size=16 \
    --num_train_epochs=300 \
    --learning_rate=1e-5 \
    --trust_remote_code
```

**Note 1:** For `--model_name`, you can choose from ["field", "adlstm", "dalle"].

**Note 2:** For `--config_path`, it is not compulsory and the default config
 will be used if not set. You can find the config files in the `configs/` folder.

### Evaluate the model

```bash
python3 eval.py \
    --model_name <model_name> \
    --pretrained_path <output_dir> \
    --dataset_name_or_path <dataset_name> \
    --output_dir <eval_output_dir> \
    --eval_batch_size=16 \
    --trust_remote_code
```

It will generate evaluation results as below:

```

```

## Download Pretrained Models

You can also download the pretrained models from the links below:

| Model | Dataset | PSNR↑ | SSIM↑ | LPIPS↓| Hugging Face |
|-------|---------|-------|-------|-------|--------------|
| field | Cube | 21.45 | 0.65  | 0.45   | [field-cube](https://huggingface.co/ziyuwwang/field-imagenet-256) |
| field | Room | 21.45 | 0.65  | 0.45   | [field-room](https://huggingface.co/ziyuwwang/field-imagenet-256) |
| field | Corridor | 21.45 | 0.65  | 0.45   | [field-corridor](https://huggingface.co/ziyuwwang/field-imagenet-256) |