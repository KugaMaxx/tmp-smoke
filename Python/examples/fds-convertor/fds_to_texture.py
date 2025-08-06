#! /usr/bin/env python3

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool


def extract_time(file_name):
    # extract time using regular expressions 
    match = re.match(r".*_(\d+)_(\d+)p(\d+)", file_name)
    if match:
        meshes = int(match.group(1))
        seconds = int(match.group(2))
        milliseconds = int(match.group(3))
        return (meshes, seconds, milliseconds)
    return (0, 0, 0)  # defualt case if no match is found


def get_history_data(args, devc_data, current_file_id):
    """
    Get historical sensor data with padding if necessary.
    
    :param devc_data: DataFrame containing the sensor data
    :param current_file_id: Current time step index
    :return: DataFrame with historical data, padded if necessary
    """
    # Filter data by sensor type
    sensor_data = devc_data.filter(like=args.selected_sensor_type)
    
    # Calculate the start index for historical data
    start_id = max(0, current_file_id - args.history_length)
    end_id = current_file_id
    
    # Get the historical data
    history_data = sensor_data.iloc[start_id:end_id]
    
    # If we don't have enough historical data, pad with the earliest available data
    if len(history_data) < args.history_length:
        needed_rows = args.history_length - len(history_data)
        # Use the first row for padding
        first_row = sensor_data.iloc[0:1]
        # Repeat the first row to fill the gap
        padding_data = pd.concat([first_row] * needed_rows, ignore_index=True)
        # Concatenate padding data with actual historical data
        history_data = pd.concat([padding_data, history_data], ignore_index=True)
    
    return history_data


def plot3d_to_flipbook(
    args,
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

    # Read the quantity data
    plot3d_data = {}
    for quantity in plot3d_quantity:
        # update data
        plot3d_data[quantity] = np.frombuffer(
            hex_data[idx:idx + 4 * NXP * NYP * NZP],
            dtype='f4'
        ).reshape(NZP, NYP, NXP).transpose(1, 2, 0)  # [z, y, x] -> [x, y, z]

        # update the index
        idx += 4 * NXP * NYP * NZP

    # Select the specified quantity
    if args.selected_quantity not in plot3d_data:
        raise ValueError(f"Selected quantity '{args.selected_quantity}' not found in plot3d data.")

    # Convert to a 2D image
    image_data = np.clip(plot3d_data[args.selected_quantity], args.min_value, args.max_value)
    image_data = (image_data * 255.0).astype(np.uint8)

    # padding images
    images = [image_data[:, :, i] for i in range(NZP)]

    num_images = args.num_rows * args.num_cols
    if len(images) < num_images:
        # print(f"Warning: num_rows * num_cols ({args.num_rows * args.num_cols}) is greater than NZP ({NZP}). "
        #       "Will add empty slices to fit.")
        # Add empty slices to fit the required number of images
        images += [np.zeros_like(images[0])] * (num_images - len(images))

    elif len(images) > num_images:
        # print(f"Warning: num_rows * num_cols ({args.num_rows * args.num_cols}) is less than NZP ({NZP}). "
        #       "Will delete some slices to fit.")
        # Truncate the list to fit the required number of images
        images = images[:num_images]

    # convert to flipbook image
    rows = []
    for i in range(0, num_images, args.num_rows):
        row = np.hstack(images[i:i + args.num_rows])
        rows.append(row)

    return np.vstack(rows)


def process_single_case(case_info):
    """
    Process a single case (train or validation) and return captions.
    
    :param case_info: Tuple containing (args, case_path, output_path, split_type)
    :return: List of caption dictionaries
    """
    args, case_path, output_path, split_type = case_info
    
    if not case_path.is_dir():
        return []
    
    # Extract case info
    case_name = case_path.stem
    devc_data = pd.read_csv(f'{case_path / (case_name + "_devc.csv")}', skiprows=1)
    q_files = sorted(case_path.glob('*.q'), key=lambda x: extract_time(x.name))
    
    # Prepare image folder
    image_path = output_path / case_name
    image_path.mkdir(parents=True, exist_ok=True)
    
    captions = []
    for file_id, q_file in enumerate(q_files):
        # Skip files that are not in the specified interval
        if file_id % args.interval != 0:
            continue
            
        # Prepare image info
        image_name = f"{case_name}_{file_id:06d}.png"
        
        # Save image
        plot3d_img = plot3d_to_flipbook(args, q_file)
        plot3d_img = Image.fromarray(plot3d_img)
        plot3d_img.save(image_path / image_name)
        
        # Get history data
        history_devc = get_history_data(args, devc_data, file_id)
        
        captions.append({
            "image": f"{split_type}/{image_path.stem}/{image_name}",
            "text": ";".join([",".join([f"{val:.3f}" for val in row]) for row in history_devc.T.values])
        })

    with open(f"{image_path / (case_name + '.jsonl')}", 'w') as f:
        for caption in captions:
            caption['image'] = caption['image'].split('/')[-1]
            json.dump(caption, f)
            f.write("\n")
    
    return captions


def process_cases_parallel(args, cases, output_path, split_type):
    """
    Process multiple cases in parallel.
    
    :param args: Arguments object
    :param cases: List of case paths
    :param output_path: Output directory path
    :param split_type: 'train' or 'validation'
    :return: List of all captions
    """
    # Prepare case info for multiprocessing
    case_infos = [(args, case, output_path, split_type) for case in cases]
    
    # Process cases in parallel
    all_captions = []
    if args.num_workers <= 1:
        # Sequential processing for debugging
        for case_info in tqdm(case_infos, desc=f"Processing {split_type} cases"):
            captions = process_single_case(case_info)
            all_captions.extend(captions)
    else:
        # Parallel processing
        with Pool(processes=args.num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_case, case_infos),
                total=len(case_infos),
                desc=f"Processing {split_type} cases"
            ))
        
        # Flatten results
        for captions in results:
            all_captions.extend(captions)
    
    return all_captions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert FDS data to LoRa format.')
    parser.add_argument(
        "--input_fds_dir",
        default=None,
        type=str,
        required=True,
        help="Path to the FDS simulation data directory containing train and validation subfolders."
    )
    parser.add_argument(
        "--output_dataset_dir",
        default=None,
        type=str,
        required=True,
        help="Path to the output directory where texture data will be saved."
    )
    parser.add_argument(
        "--plot3d_quantity",
        default=[
            'OPTICAL DENSITY',
            'U-VELOCITY',
            'V-VELOCITY',
            'W-VELOCITY',
            'TEMPERATURE'
        ],
        nargs='+',
        help=(
            "List of quantities to extract from the plot3d data. "
            "Please check &DUMP PLOT3D_QUANTITY in fds file to find the corresponding list."
        )
    )
    parser.add_argument(
        "--selected_quantity",
        default="OPTICAL DENSITY",
        type=str,
        help=(
            "Specific quantity to extract from the plot3d_quantity list for visualization."
        ),
    )
    parser.add_argument(
        "--selected_sensor_type",
        default="SD",
        type=str,
        help=(
            "which sensor's data will be extracted. "
            "HD for temperature sensors, SD for smoke sensors.",
        )
    )
    parser.add_argument(
        "--history_length",
        default=128,
        type=int,
        help="Length of historical sensor data to use.",
    )
    parser.add_argument(
        "--interval",
        default=1,
        type=int,
        help="Interval for extracting data from the plot3d file.",
    )

    # flipbook parameters
    parser.add_argument(
        "--num_rows",
        default=2,
        type=int,
        help="Number of x-axis slice data in the flipbook.",
    )
    parser.add_argument(
        "--num_cols",
        default=15,
        type=int,
        help="Number of y-axis slice data in the flipbook.",
    )
    parser.add_argument(
        "--min_value",
        default=0.0,
        type=float,
        help="Minimum value for the texture data.",
    )
    parser.add_argument(
        "--max_value",
        default=1.5,
        type=float,
        help="Maximum value for the texture data.",
    )
    parser.add_argument(
        "--num_workers",
        default=0,
        type=int,
        help="Number of CPU cores to use for parallel processing. Set 0 will be sequential processing.",
    )
    args = parser.parse_args()

    # fds simulation data directory
    args.input_fds_dir = Path(args.input_fds_dir)

    # create output directory
    args.output_dataset_dir = Path(args.output_dataset_dir)
    args.output_dataset_dir.mkdir(parents=True, exist_ok=True)

    # create train subfolder
    dataset_train_path = args.output_dataset_dir / 'train'
    dataset_train_path.mkdir(parents=True, exist_ok=True)
    dataset_validation_path = args.output_dataset_dir / 'validation'
    dataset_validation_path.mkdir(parents=True, exist_ok=True)

    # make train subfolder
    print("Making train subfolder...")
    train_cases = list(sorted((args.input_fds_dir / 'train').glob('*')))
    train_captions = process_cases_parallel(args, train_cases, dataset_train_path, 'train')


    # make validation subfolder
    print("Making validation subfolder...")
    validation_cases = list(sorted((args.input_fds_dir / 'validation').glob('*')))
    validation_captions = process_cases_parallel(args, validation_cases, dataset_validation_path, 'validation')

    # generate huggingface dataset format
    os.system("cp "
              f"{str((Path(__file__).parent / 'misc' / 'template.py').resolve())} "
              f"{str(args.output_dataset_dir / (str(args.output_dataset_dir.stem) + '.py'))}")
