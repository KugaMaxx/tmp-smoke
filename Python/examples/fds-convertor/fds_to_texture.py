#! /usr/bin/env python3

import os
import re
import math
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def extract_time(file_name):
    # extract time using regular expressions 
    match = re.match(r".*_(\d+)_(\d+)p(\d+)", file_name)
    if match:
        meshes = int(match.group(1))
        seconds = int(match.group(2))
        milliseconds = int(match.group(3))
        return (meshes, seconds, milliseconds)
    return (0, 0, 0)  # defualt case if no match is found


def plot3d_to_flipbook(
    qfile,
    plot3d_quantity=[
        'OPTICAL DENSITY',
        'U-VELOCITY',
        'V-VELOCITY',
        'W-VELOCITY',
        'TEMPERATURE'
    ]
):
    """
    Read data from plot3d file.
    :param qfile: File path
    :param plot3d_quantity: List of quantities to extract from the plot3d file.
    :return: A dictionary with quantities as keys and flipbook images as values.
    """
    idx = 0
    with open(qfile, 'rb') as file:
        hex_data = file.read()

    # Read basic info
    TMP, NXP, NYP, NZP = np.frombuffer(hex_data[idx:idx + 4 * 4], dtype='i4')
    idx += 4 * 4

    D1, D2, D3, D4 = np.frombuffer(hex_data[idx:idx + 4 * 4], dtype='S4')
    idx += 4 * 4

    D5, D6, D7, D8 = np.frombuffer(hex_data[idx:idx + 4 * 4], dtype='S4')
    idx += 4 * 4

    size_z = math.ceil(math.sqrt(NZP))

    # Read the quantity data
    flipbook = {}
    for quantity in plot3d_quantity:
        # update data
        plot3d_data = np.frombuffer(
            hex_data[idx:idx + 4 * NXP * NYP * NZP],
            dtype='f4'
        ).reshape(NZP, NYP, NXP).transpose(2, 1, 0)  # [z, y, x] -> [x, y, z]

        # update the index
        idx += 4 * NXP * NYP * NZP

        # normalize
        image_data = np.clip(plot3d_data, 0, 1.5)
        image_data = (image_data * 255.0).astype(np.uint8)

        # padding images
        images = [image_data[:, :, i] for i in range(NZP)]
        if len(images) < size_z ** 2:
            # pad with zeros if not enough images
            pad_size = size_z ** 2 - len(images)
            images += [np.zeros_like(images[0])] * pad_size

        # convert to flipbook image
        rows = []
        for i in range(0, size_z ** 2, size_z):
            row = np.hstack(images[i:i + size_z])
            rows.append(row)
        flipbook[quantity] = np.vstack(rows)

    return flipbook


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert FDS data to LoRa format.')
    parser.add_argument('--fds_data_dir', default=str((Path(__file__).parent / '../../data/cube-fds').resolve()), type=str)
    parser.add_argument('--texture_data_dir', default=str((Path(__file__).parent / '../../data/cube-texture').resolve()), type=str)
    parser.add_argument('--quantity', default='OPTICAL DENSITY', type=str)
    parser.add_argument('--sensor_type', default='HD', type=str, help='HD for temperature sensors, SD for smoke sensors.')
    args = parser.parse_args()

    # fds simulation data directory
    args.fds_data_dir = Path(args.fds_data_dir)

    # create output directory
    args.texture_data_dir = Path(args.texture_data_dir)
    args.texture_data_dir.mkdir(parents=True, exist_ok=True)

    # read paired fds index
    with open(f'{args.fds_data_dir / 'index.jsonl'}', 'r') as f:
        index = json.load(f)

    # create train subfolder
    texture_data_database_dir = args.texture_data_dir / 'database'
    texture_data_database_dir.mkdir(parents=True, exist_ok=True)
    texture_data_train_dir = args.texture_data_dir / 'train'
    texture_data_train_dir.mkdir(parents=True, exist_ok=True)
    texture_data_validation_dir = args.texture_data_dir / 'validation'
    texture_data_validation_dir.mkdir(parents=True, exist_ok=True)

    # make train subfolder
    print("Making train subfolder...")
    captions = []
    for case_id, case_info in tqdm(enumerate(index["train"]), total=len(index["train"])):
        database_case = args.fds_data_dir / case_info['source']
        train_case = args.fds_data_dir / case_info['target']

        database_devc_data = pd.read_csv(f'{database_case / (database_case.stem + '_devc.csv')}', skiprows=1)
        train_devc_data = pd.read_csv(f'{train_case / (train_case.stem + '_devc.csv')}', skiprows=1)

        database_q_files = sorted(database_case.glob('*.q'), key=lambda x: extract_time(x.name))
        train_q_files = sorted(train_case.glob('*.q'), key=lambda x: extract_time(x.name))

        for file_id, (database_q_file, train_q_file) in enumerate(zip(database_q_files, train_q_files)):
            # convert plot3d data to flipbook image
            database_plot3d_img = plot3d_to_flipbook(database_q_file)[args.quantity]
            train_plot3d_img = plot3d_to_flipbook(train_q_file)[args.quantity]

            # save database image
            database_image_dir = texture_data_database_dir / database_case.stem
            database_image_dir.mkdir(parents=True, exist_ok=True)
            database_image_name = f"{database_case.stem}_{file_id:03d}.png"
            database_image = Image.fromarray(database_plot3d_img)
            database_image.save(database_image_dir / database_image_name)

            # save train image
            train_image_dir = texture_data_train_dir / train_case.stem
            train_image_dir.mkdir(parents=True, exist_ok=True)
            train_image_name = f"{train_case.stem}_{file_id:03d}.png"
            train_image = Image.fromarray(train_plot3d_img)
            train_image.save(train_image_dir / train_image_name)

            # residual check
            # Remove 'Time' column before calculating residuals
            database_devc = database_devc_data.drop(columns=['Time'], errors='ignore').iloc[file_id]
            train_devc = train_devc_data.drop(columns=['Time'], errors='ignore').iloc[file_id]
            residual_devc = train_devc - database_devc
            
            captions.append(
                {
                    "conditioning_image": f"database/{database_image_dir.stem}/{database_image_name}",
                    "image": f"train/{train_image_dir.stem}/{train_image_name}",
                    "text": ','.join([f"{float(devc) + 1e-2:.2f}" for devc in residual_devc]),
                }
            )

    with open(f"{texture_data_train_dir / 'prompt.jsonl'}", 'w') as f:
        for caption in captions:
            json.dump(caption, f)
            f.write("\n")

    # make validation subfolder
    print("Making validation subfolder...")
    captions = []
    for case_id, case_info in tqdm(enumerate(index["validation"]), total=len(index["validation"])):
        validation_case = args.fds_data_dir / case_info['target']
        validation_devc_data = pd.read_csv(f'{validation_case / (validation_case.stem + '_devc.csv')}', skiprows=1)
        validation_q_files = sorted(validation_case.glob('*.q'), key=lambda x: extract_time(x.name))

        for file_id, validation_q_file in enumerate(validation_q_files):
            # convert plot3d data to flipbook image
            validation_plot3d_img = plot3d_to_flipbook(validation_q_file)[args.quantity]

            # save validation image
            validation_image_dir = texture_data_validation_dir / validation_case.stem
            validation_image_dir.mkdir(parents=True, exist_ok=True)
            validation_image_name = f"{validation_case.stem}_{file_id:03d}.png"
            validation_image = Image.fromarray(validation_plot3d_img)
            validation_image.save(validation_image_dir / validation_image_name)

    # generate huggingface dataset format
    os.system("cp "
              f"{str((Path(__file__).parent / 'misc' / 'template.py').resolve())} "
              f"{str(args.texture_data_dir / (str(args.texture_data_dir.stem) + '.py'))}")
