import re
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm


def extract_time(file_name):
    # extract time using regular expressions 
    match = re.match(r".*_(\d+)_(\d+)p(\d+)", file_name)
    if match:
        meshes = int(match.group(1))
        seconds = int(match.group(2))
        milliseconds = int(match.group(3))
        return (meshes, seconds, milliseconds)
    return (0, 0, 0)  # defualt case if no match is found


def load_plot3d_data(
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
    :param plot3d_quantity: List of quantities to read
    :return: Dictionary of plot3d data
    """
    plot3d_data, idx = {}, 0
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
    for quantity in plot3d_quantity:
        # update data
        plot3d_data[quantity] = np.frombuffer(
            hex_data[idx:idx + 4 * NXP * NYP * NZP],
            dtype='f4'
        ).reshape(NZP, NYP, NXP).transpose(2, 1, 0)  # [z, y, x] -> [x, y, z]

        # update the index
        idx += 4 * NXP * NYP * NZP

    return plot3d_data


def plot3d_to_flipbook(
    plot3d_data,
    image_size=(512, 512),
    quantity='OPTICAL DENSITY'
):
    """
    Convert plot3d data to image.
    :param plot3d_data: Plot3d data
    :param image_size: Image size
    :param quantity: Quantity to convert
    :return: Image
    """
    # convert plot3d data to image
    density = plot3d_data[quantity].transpose(2, 1, 0).astype(np.float32)
    density[density > 1.0] = 1.0
    density = (density * 255.0).astype(np.uint8)

    # 将原始数组沿Z轴（轴2）切分成64张 (64, 64) 的图片
    images = [density[i, :, :] for i in range(64)]

    # 将这些图片按照 8x8 排列成一张大图
    # 我们需要使用 `np.vstack` 和 `np.hstack` 来拼接图片
    rows = []
    for i in range(0, 64, 8):
        row = np.hstack(images[i:i+8])  # 每行拼接8张图片
        rows.append(row)

    image = np.vstack(rows)

    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert FDS data to LoRa format.')
    parser.add_argument('--fds_data_dir', default='/home/dszh/workspace/tmp-smoke/Python/data/cube-fds', type=str)
    parser.add_argument('--texture_data_dir', default='/home/dszh/workspace/tmp-smoke/Python/data/cube-texture', type=str)
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
    texture_data_train_dir = args.texture_data_dir / 'train'
    texture_data_train_conditioning_image_dir = texture_data_train_dir / 'conditioning_image'
    texture_data_train_conditioning_image_dir.mkdir(parents=True, exist_ok=True)
    texture_data_train_image_dir = texture_data_train_dir / 'image'
    texture_data_train_image_dir.mkdir(parents=True, exist_ok=True)

    print("Making train subfolder...")
    captions = []
    for case_id, case_info in tqdm(enumerate(index["train"])):
        source_case = args.fds_data_dir / case_info['source']
        target_case = args.fds_data_dir / case_info['target']

        source_devc_data = pd.read_csv(f'{source_case / (source_case.stem + '_devc.csv')}', skiprows=1)
        source_devc_data = source_devc_data.filter(like='HD')
        target_devc_data = pd.read_csv(f'{target_case / (target_case.stem + '_devc.csv')}', skiprows=1)
        target_devc_data = target_devc_data.filter(like='HD')

        source_q_files = sorted(source_case.glob('*.q'), key=lambda x: extract_time(x.name))
        target_q_files = sorted(target_case.glob('*.q'), key=lambda x: extract_time(x.name))

        for source_q_file, target_q_file in zip(source_q_files, target_q_files):
            source_plot3d_data = load_plot3d_data(source_q_file)
            target_plot3d_data = load_plot3d_data(target_q_file)

            source_plot3d_img = plot3d_to_flipbook(source_plot3d_data, quantity='OPTICAL DENSITY')
            target_plot3d_img = plot3d_to_flipbook(target_plot3d_data, quantity='OPTICAL DENSITY')

            # 上面回头合并成一个

        for file_id, (source_density_img, target_density_img) in enumerate(zip(source_plot3d_imgs, target_plot3d_imgs)):
            # residual check
            source_devc = source_devc_data.iloc[file_id]
            target_devc = target_devc_data.iloc[file_id]
            resiual_devc = target_devc - source_devc

            # save source image
            source_image_dir = source_image_path / f"{source_case.stem}_{file_id:03d}.png"
            source_image = Image.fromarray(source_density_img)
            source_image.save(source_image_dir)

            # save target image
            target_image_dir = target_image_path / f"{source_case.stem}_{file_id:03d}.png"
            target_image = Image.fromarray(target_density_img)
            target_image.save(target_image_dir)

            captions.append(
                {
                    "conditioning_image": f"conditioning_image/{source_image_dir.name}",
                    "image": f"image/{target_image_dir.name}",
                    "text": ','.join([f"{devc:.2f}" for devc in resiual_devc]),
                }
            )
        
        # update
        step += 1
        progress_bar.update(1)

    with open(f"{args.texture_data_dir / 'prompt.jsonl'}", 'w') as f:
        for caption in captions:
            json.dump(caption, f)
            f.write("\n")
