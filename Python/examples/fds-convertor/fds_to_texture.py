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


def get_history_data(devc_data, current_file_id, history_length, sensor_type='HD'):
    """
    Get historical sensor data with padding if necessary.
    
    :param devc_data: DataFrame containing the sensor data
    :param current_file_id: Current time step index
    :param history_length: Number of historical time steps to retrieve
    :param sensor_type: Type of sensor to filter ('HD' for temperature sensors, 'SD' for smoke sensors)
    :return: DataFrame with historical data, padded if necessary
    """
    # Filter data by sensor type
    sensor_data = devc_data.filter(like=sensor_type)
    
    # Calculate the start index for historical data
    start_id = max(0, current_file_id - history_length)
    end_id = current_file_id
    
    # Get the historical data
    history_data = sensor_data.iloc[start_id:end_id]
    
    # If we don't have enough historical data, pad with the earliest available data
    if len(history_data) < history_length:
        needed_rows = history_length - len(history_data)
        # Use the first row for padding
        first_row = sensor_data.iloc[0:1]
        # Repeat the first row to fill the gap
        padding_data = pd.concat([first_row] * needed_rows, ignore_index=True)
        # Concatenate padding data with actual historical data
        history_data = pd.concat([padding_data, history_data], ignore_index=True)
    
    return history_data


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
    parser.add_argument('--fds_data_path', default=str((Path(__file__).parent / '../../data/cube-fds').resolve()), type=str)
    parser.add_argument('--texture_data_path', default=str((Path(__file__).parent / '../../data/cube-texture').resolve()), type=str)
    parser.add_argument('--quantity', default='OPTICAL DENSITY', type=str)
    parser.add_argument('--sensor_type', default='HD', type=str, help='HD for temperature sensors, SD for smoke sensors.')
    parser.add_argument('--history_length', default=32, type=int, help='Length of historical sensor data to use.')
    args = parser.parse_args()

    # fds simulation data directory
    args.fds_data_path = Path(args.fds_data_path)

    # create output directory
    args.texture_data_path = Path(args.texture_data_path)
    args.texture_data_path.mkdir(parents=True, exist_ok=True)

    # read paired fds index
    with open(f'{args.fds_data_path / 'index.jsonl'}', 'r') as f:
        index = json.load(f)

    # create train subfolder
    texture_data_database_path = args.texture_data_path / 'database'
    texture_data_database_path.mkdir(parents=True, exist_ok=True)
    texture_data_train_path = args.texture_data_path / 'train'
    texture_data_train_path.mkdir(parents=True, exist_ok=True)
    texture_data_validation_path = args.texture_data_path / 'validation'
    texture_data_validation_path.mkdir(parents=True, exist_ok=True)

    # make train subfolder
    print("Making train subfolder...")
    captions = []
    for case_id, case_info in tqdm(enumerate(index["train"]), total=len(index["train"])):            
        database_case = args.fds_data_path / case_info['source']
        train_case = args.fds_data_path / case_info['target']

        database_devc_data = pd.read_csv(f'{database_case / (database_case.stem + '_devc.csv')}', skiprows=1)
        train_devc_data = pd.read_csv(f'{train_case / (train_case.stem + '_devc.csv')}', skiprows=1)

        database_q_files = sorted(database_case.glob('*.q'), key=lambda x: extract_time(x.name))
        train_q_files = sorted(train_case.glob('*.q'), key=lambda x: extract_time(x.name))

        for file_id, (database_q_file, train_q_file) in enumerate(zip(database_q_files, train_q_files)):
            # prepare database image folder
            database_image_path = texture_data_database_path / database_case.stem
            database_image_path.mkdir(parents=True, exist_ok=True)
            database_image_name = f"{database_case.stem}_{file_id:03d}.png"

            # prepare train image folder
            train_image_path = texture_data_train_path / train_case.stem
            train_image_path.mkdir(parents=True, exist_ok=True)
            train_image_name = f"{train_case.stem}_{file_id:03d}.png"

            # # save database image
            # database_plot3d_img = plot3d_to_flipbook(database_q_file)[args.quantity]
            # database_image = Image.fromarray(database_plot3d_img)
            # database_image.save(database_image_path / database_image_name)

            # # save train image
            # train_plot3d_img = plot3d_to_flipbook(train_q_file)[args.quantity]
            # train_image = Image.fromarray(train_plot3d_img)
            # train_image.save(train_image_path / train_image_name)

            # residual check
            # Remove 'Time' column before calculating residuals
            database_devc = database_devc_data.iloc[file_id]
            train_devc = train_devc_data.iloc[file_id]

            history_devc = get_history_data(train_devc_data, file_id, args.history_length, args.sensor_type)

            captions.append(
                {
                    "conditioning_image": f"database/{database_image_path.stem}/{database_image_name}",
                    "image": f"train/{train_image_path.stem}/{train_image_name}",
                    "text": ";".join([",".join(map(str, row)) for row in history_devc.T.values])
                }
            )

    with open(f"{texture_data_train_path / 'prompt.jsonl'}", 'w') as f:
        for caption in captions:
            json.dump(caption, f)
            f.write("\n")

    # make validation subfolder
    print("Making validation subfolder...")
    captions = []
    for case_id, case_info in tqdm(enumerate(index["validation"]), total=len(index["validation"])):
        validation_case = args.fds_data_path / case_info['target']
        validation_devc_data = pd.read_csv(f'{validation_case / (validation_case.stem + '_devc.csv')}', skiprows=1)
        validation_q_files = sorted(validation_case.glob('*.q'), key=lambda x: extract_time(x.name))

        for file_id, validation_q_file in enumerate(validation_q_files):
            # prepare validation image folder
            validation_image_path = texture_data_validation_path / validation_case.stem
            validation_image_path.mkdir(parents=True, exist_ok=True)
            validation_image_name = f"{validation_case.stem}_{file_id:03d}.png"

            # # save validation image
            # validation_plot3d_img = plot3d_to_flipbook(validation_q_file)[args.quantity]
            # validation_image = Image.fromarray(validation_plot3d_img)
            # validation_image.save(validation_image_path / validation_image_name)

            validation_devc = validation_devc_data.iloc[file_id]
            history_devc = get_history_data(validation_devc_data, file_id, args.history_length, args.sensor_type)
            captions.append(
                {
                    "conditioning_image": f"validation/{validation_image_path.stem}/{validation_image_name}",
                    "image": f"validation/{validation_image_path.stem}/{validation_image_name}",
                    "text": ";".join([",".join(map(str, row)) for row in history_devc.T.values])
                }
            )

    with open(f"{texture_data_validation_path / 'prompt.jsonl'}", 'w') as f:
        for caption in captions:
            json.dump(caption, f)
            f.write("\n")

    # generate huggingface dataset format
    os.system("cp "
              f"{str((Path(__file__).parent / 'misc' / 'template.py').resolve())} "
              f"{str(args.texture_data_path / (str(args.texture_data_path.stem) + '.py'))}")
