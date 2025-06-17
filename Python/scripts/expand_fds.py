#!/usr/bin/env python
# This script helps 生成项目需要的 fds 文件结构从给定的建筑布局中。首先，不同的建筑基础
# 布局需要放在同一文件夹中，随后该代码会在指定的 output_dir 下生成三个子文件夹：
# 以 --xxx 间隔生成的参考 fds 数据，用于
# 以 --xxx 等间隔随机采样生成的数据，用于
# 随后在区间内随机生成的 validation 数据，用于

import re
import json
import random
import argparse
from pathlib import Path
from tqdm.auto import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Expand FDS data with varying heat release rates.')
    parser.add_argument('--fds_data_dir', default='/home/dszh/workspace/tmp-smoke/Python/data/cube-fds/.fds', type=str)
    parser.add_argument('--output_dir', default='/home/dszh/workspace/tmp-smoke/Python/data/cube-fds', type=str)
    parser.add_argument('--start_hrr', default=100, type=int)
    parser.add_argument('--end_hrr', default=2100, type=int)
    parser.add_argument('--step_hrr', default=100, type=int)
    parser.add_argument('--seed', default=123, type=int)
    args = parser.parse_args()

    # fix random seed
    if args.seed is not None:
        random.seed(args.seed)

    # fds directory
    args.fds_data_dir = Path(args.fds_data_dir)
    
    # create output directory
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_baseline_dir = args.output_dir / 'baseline'
    output_baseline_dir.mkdir(parents=True, exist_ok=True)

    output_train_dir = args.output_dir / 'train'
    output_train_dir.mkdir(parents=True, exist_ok=True)

    output_validation_dir = args.output_dir / 'validation'
    output_validation_dir.mkdir(parents=True, exist_ok=True)

    # set fds update function
    def modify_hrr_and_save(fds_file, hrr, output_dir, output_name):
        # read existing fds file
        with open(fds_file, 'r', encoding='utf-8') as f:
            fds_info = f.read()
            fds_info = re.sub(r'HRRPUA=\d+\.?\d*', f'HRRPUA={hrr:.1f}', fds_info)

        # create new subfolder
        output_dir = output_dir / (output_name + f'_h{hrr:04d}')
        output_dir.mkdir(parents=True, exist_ok=True)

        # rewrite
        output_fds_file = output_dir / (output_dir.name + '.fds')
        with open(output_fds_file, 'w', encoding='utf-8') as f:
            f.write(fds_info)

        return output_fds_file
    
    # look through all fds files
    captions = {'train': [], 'validation': []}
    for fds_case in tqdm(sorted(args.fds_data_dir.glob('*'))):
        fds_file = fds_case / (fds_case.stem + '.fds')
        fds_file_name = fds_case.name
        
        # create train
        for hrr in range(args.start_hrr, args.end_hrr, args.step_hrr):

            # save basline
            baseline_fds_file = modify_hrr_and_save(fds_file, hrr, output_dir=output_baseline_dir, output_name=fds_file_name)

            half_step_hrr = args.step_hrr / 2.0
            hrr = hrr + random.randint(-int(half_step_hrr), int(half_step_hrr))

            # save sampling
            train_fds_file = modify_hrr_and_save(fds_file, hrr, output_dir=output_train_dir, output_name=fds_file_name)

            captions['train'].append(
                {
                    "source": f"{output_baseline_dir.stem}/{baseline_fds_file.stem}",
                    "target": f"{output_train_dir.stem}/{train_fds_file.stem}",
                }
            )

        # create validation
        hrr = random.randint(args.start_hrr, args.end_hrr)

        # save validation
        validation_fds_file = modify_hrr_and_save(fds_file, hrr, output_dir=output_validation_dir, output_name=fds_file_name)
        captions['validation'].append(
            {
                "source": f"{output_validation_dir.stem}/{validation_fds_file.stem}",
                "target": None,
            }
        )
        
    with open(f"{args.output_dir / 'index.jsonl'}", 'w', encoding='utf-8') as f:
        for split in captions:
            for caption in captions[split]:
                json.dump(caption, f, ensure_ascii=False)
                f.write("\n")
