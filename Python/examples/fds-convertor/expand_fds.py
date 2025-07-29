#!/usr/bin/env python3
# This script helps expand and generate the FDS file required for the project
# from given building layouts and fire location.
#
# Different basic building layouts need to be placed in the --fds_case_dir,
# then this code will generate various FDS files as train with different heat
# release rates (HRR) based on the given --start_hrr, --end_hrr, and --step_hrr.
#
# Then it will also generate a validation FDS file with a random HRR
# within the specified range.

import re
import json
import random
import argparse
from pathlib import Path
from tqdm.auto import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Expand FDS data with varying heat release rates.')
    parser.add_argument(
        "--fds_case_dir",
        default=str((Path(__file__).parent / "../../data/corridor-fds/.fds").resolve()),
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        default=str((Path(__file__).parent / "../../data/corridor-fds").resolve()),
        type=str,
    )
    parser.add_argument("--start_hrr", default=100, type=int)
    parser.add_argument("--end_hrr", default=2100, type=int)
    parser.add_argument("--step_hrr", default=100, type=int)
    parser.add_argument("--seed", default=123, type=int)
    args = parser.parse_args()

    # fix random seed
    if args.seed is not None:
        random.seed(args.seed)

    # predefined fds case directory
    args.fds_case_dir = Path(args.fds_case_dir)

    # create output directory
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_train_dir = args.output_dir / 'train'
    output_train_dir.mkdir(parents=True, exist_ok=True)

    output_validation_dir = args.output_dir / 'validation'
    output_validation_dir.mkdir(parents=True, exist_ok=True)

    # set fds update function
    def modify_hrr_and_save(fds_file, hrr, output_dir, output_name):
        # read existing fds file
        with open(fds_file, 'r', encoding='utf-8') as f:
            fds_info = f.read()

        # create new subfolder
        output_name = output_name + f'_h{hrr:04d}'
        output_dir = output_dir / output_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # modify
        fds_info = re.sub(r"HRRPUA=\d+\.?\d*", f"HRRPUA={hrr:.1f}", fds_info)
        fds_info = re.sub(r"CHID='([^']*)'", f"CHID='{output_name}'", fds_info)

        # rewrite
        output_fds_file = output_dir / (output_dir.name + '.fds')
        with open(output_fds_file, 'w', encoding='utf-8') as f:
            f.write(fds_info)

        return output_fds_file

    # look through all fds files
    for fds_case in tqdm(sorted(args.fds_case_dir.glob('*'))):
        fds_file = fds_case / (fds_case.stem + '.fds')
        fds_file_name = fds_case.name

        # create train
        for hrr in range(args.start_hrr, args.end_hrr, args.step_hrr):

            # save train
            modify_hrr_and_save(fds_file, hrr, output_dir=output_train_dir, output_name=fds_file_name)

        # create validation
        hrr = random.randint(args.start_hrr, args.end_hrr)

        # save validation
        validation_fds_file = modify_hrr_and_save(fds_file, hrr, output_dir=output_validation_dir, output_name=fds_file_name)
