import torch
import faiss
import openvdb
import argparse
import numpy as np
import pandas as pd
import pyvista as pv
from PIL import Image
from pathlib import Path
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionImg2ImgPipeline

from matplotlib.colors import ListedColormap


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert FDS data to LoRa format.')
    parser.add_argument('--dataset_path', default='/home/dszh/workspace/data/Cube-v4', type=str)
    parser.add_argument('--validation_dir', default='/home/dszh/workspace/tmp-smoke/Python/data/cube-fds/validation/cube_s01_h0602', type=str)
    args = parser.parse_args()

    # create output directory
    dataset_path = Path(args.dataset_path)

    # initialize text encoder
    text_encoder = CLIPTextModel.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="text_encoder"
    ).to("cuda")
    text_encoder.requires_grad_(False)

    tokenizer = CLIPTokenizer.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="tokenizer"
    )

    cases = [
        f for f in sorted(dataset_path.glob('*'))
        if f.is_dir() and not f.name.startswith('.')
    ]
    embeds, database = [], []

    # for case_id, case in enumerate(cases):
    #     if 'Cube04_S01' not in case.name: continue
    #     print(case.name)

    #     devc_list = pd.read_csv(f"{case / (case.stem + '_devc.csv')}", skiprows=1)
    #     # devc_list = devc_list.filter(like='HD')
        
    #     for devc_id, devc_data in devc_list.iterrows():
    #         caption = ','.join([f"{data:.2f}" for data in devc_data.values])
    #         input = tokenizer(
    #             caption, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    #         ).input_ids.to("cuda")
    #         embed = text_encoder(input, return_dict=False)[0]

    #         embeds.append(embed.flatten().cpu().numpy())
    #         database.append(
    #             {
    #                 'name': case.stem,
    #                 'idx': devc_id,
    #                 'devc': devc_data,
    #                 'image_path': Path('/home/dszh/workspace/data/csmv-cube-v4-rag-residual/image') / f"{case.stem}_{devc_id:03d}.png"
    #             }
    #         )

    # # make faiss index
    # data = np.array(embeds)
    # index = faiss.IndexFlatL2(data.shape[1])
    # index.add(data)

    # initialize model
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
        "/home/dszh/workspace/diffusers/examples/text_to_image/output-image-to-image", 
        torch_dtype=torch.bfloat16, 
        safety_checker=None,
    ).to("cuda")

    pipeline.unet.to(memory_format=torch.channels_last)
    pipeline.vae.to(memory_format=torch.channels_last)


    # args.validation_dir = Path(args.validation_dir)
    # devc_list = pd.read_csv(f"{args.validation_dir / (args.validation_dir.stem + '_devc.csv')}", skiprows=1)
    # # devc_list = devc_list.filter(like='HD')
    
    # ====
    devc_list = pd.read_csv(f"/home/dszh/workspace/data/Cube-v4/Cube04_S01_H0600/Cube04_S01_H0600_devc.csv", skiprows=1)
    # devc_list = devc_list.filter(like='HD')
    
    for devc_id, devc_data in devc_list.iterrows():
        caption = ','.join([f"{data:.2f}" for data in devc_data.values])
        input = tokenizer(
            caption, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids.to("cuda")
        embed = text_encoder(input, return_dict=False)[0]

        embeds.append(embed.flatten().cpu().numpy())
        database.append(
            {
                'name': 'Cube04_S01_H0600',
                'idx': devc_id,
                'devc': devc_data,
                'image_path': Path('/home/dszh/workspace/data/csmv-cube-v4-rag-residual/image') / f"Cube04_S01_H0600_{devc_id:03d}.png"
            }
        )
    # ====

    frames = []
    for devc_id, devc_data in devc_list.iterrows():
        if devc_id >= 240: break

        caption = ','.join([f"{data:.2f}" for data in devc_data.values])
        input = tokenizer(
            caption, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids.to("cuda")
        search_embed = text_encoder(input, return_dict=False)[0]
        search_embed = search_embed.flatten().unsqueeze(0).cpu().numpy()

        # D, I = index.search(search_embed, 1)

        # retrieved_data = [database[i] for i in I[0]]

        retrieved_data = [database[devc_id]]

        print(caption)
        print(retrieved_data)
        print('====================================')

        image = Image.open(retrieved_data[0]['image_path']).convert("RGB")
        devc_res = devc_data - retrieved_data[0]['devc']
        prompt = ','.join([f"{data:.2f}" for data in devc_res.values])

        image = pipeline(prompt, image, num_inference_steps=20, generator=torch.manual_seed(42)).images[0].convert('L')

        # 如果是flipbook类型
        IJK = (64, 64, 64)
        X, Y, Z = np.mgrid[0:IJK[0], 0:IJK[1], 0:IJK[2]]

        X = X.astype(np.float32)
        Y = Y.astype(np.float32)
        Z = Z.astype(np.float32)
        image = np.array(image)

        # 按行和列拆分大图
        tiles = []
        for i in range(0, image.shape[0], 64):
            row = image[i:i+64, :]  # 取一行（包含 8 张小图）
            for j in range(0, image.shape[1], 64):
                # 提取每个小图
                small_image = row[:, j:j+64]
                tiles.append(small_image)

        # 将小图列表转换为 3D 数组
        density = np.stack(tiles, axis=0).astype(np.float32) / 255.0  # 形状为 (64, 64, 64)

        # # export to VDB
        # vdb_grid = openvdb.FloatGrid()
        # vdb_grid.copyFromArray(density)
        # vdb_grid.name = "density"
        # openvdb.write(f"/home/dszh/workspace/tmp-smoke/Python/output/{devc_id}.vdb", grids=[vdb_grid])

        # volume rendering
        grid = pv.StructuredGrid(X, Y, Z)
        grid.point_data["scalars"] = density.flatten()

        # Define the colors we want to use
        colors = np.ones((256, 4))
        colors[:, 0] = 0.0
        colors[:, 1] = 0.0
        colors[:, 2] = 0.0
        colors[:, 3] = np.linspace(0, 0.1, 256)
        cmap = ListedColormap(colors)

        plotter = pv.Plotter(off_screen=True)
        plotter.add_volume(grid, scalars="scalars", opacity='linear', cmap=cmap, clim=[0, 5])
        plotter.screenshot(f"/home/dszh/workspace/tmp-smoke/Python/output/{devc_id}.png")
        plotter.close()

        frames.append(Image.open(f"/home/dszh/workspace/tmp-smoke/Python/output/{devc_id}.png"))

    frames[0].save(
        "/home/dszh/workspace/tmp-smoke/Python/output/animation.gif",
        save_all=True,
        append_images=frames[1:],
        duration=200,  # 每帧200ms，可调整
        loop=0
    )
