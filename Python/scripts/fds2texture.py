import json
import argparse
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm

from scripts.misc import extract_time
from scripts.common import load_plot3d_data, plot3d_to_image, plot3d_to_flipbook

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert FDS data to LoRa format.')
    parser.add_argument('--dataset_path', default='/home/dszh/workspace/tmp-smoke/Python/data/cube-fds', type=str)
    parser.add_argument('--output_dir', default='/home/dszh/workspace/tmp-smoke/Python/data/cube-texture', type=str)
    args = parser.parse_args()

    # create output directory
    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # create subdirectories for images and conditioning images
    source_image_path = output_dir / 'conditioning_image'
    source_image_path.mkdir(parents=True, exist_ok=True)
    target_image_path = output_dir / 'image'
    target_image_path.mkdir(parents=True, exist_ok=True)

    cases = [
        f for f in sorted(dataset_path.glob('*'))
        if f.is_dir() and not f.name.startswith('.')
    ]
    captions = []
    
    # initialize progress bar
    step = 0
    progress_bar = tqdm(
        range(0, len(cases)),
        initial=step,
        desc="Process"
    )

    for case_id, case in enumerate(cases):
        if '0500' in case.stem:
            continue
        else:
            source_case = cases[case_id - 1]
            target_case = cases[case_id]

        source_files = sorted(source_case.glob('*.q'), key=lambda x: extract_time(x.name))
        target_files = sorted(target_case.glob('*.q'), key=lambda x: extract_time(x.name))

        source_plot3d_data = [load_plot3d_data(file) for file in source_files]
        target_plot3d_data = [load_plot3d_data(file) for file in target_files]

        source_devc_list = pd.read_csv(f'{source_case / (source_case.stem + '_devc.csv')}', skiprows=1)
        source_devc_list = source_devc_list.filter(like='HD')
        target_devc_list = pd.read_csv(f'{target_case / (target_case.stem + '_devc.csv')}', skiprows=1)
        target_devc_list = target_devc_list.filter(like='HD')

        source_density_imgs = [plot3d_to_flipbook(data, quantity='OPTICAL DENSITY') for data in source_plot3d_data]
        target_density_imgs = [plot3d_to_flipbook(data, quantity='OPTICAL DENSITY') for data in target_plot3d_data]

        for file_id, (source_density_img, target_density_img) in enumerate(zip(source_density_imgs, target_density_imgs)):
            # residual check
            source_devc = source_devc_list.iloc[file_id]
            target_devc = target_devc_list.iloc[file_id]
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

    with open(f"{output_dir / 'prompt.jsonl'}", 'w') as f:
        for caption in captions:
            json.dump(caption, f)
            f.write("\n")
