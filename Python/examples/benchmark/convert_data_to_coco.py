#!/usr/bin/env python3
"""
Convert demo dataset to COCO format required by DF-GAN

Conversion workflow:
1. Read demo dataset parquet files
2. Save images as JPEG to images/train2014 and val2014 directories
3. Convert text field to caption text files
4. Create filenames.pickle file
5. Generate captions_DAMSM.pickle file (simplified version)
"""

import os
import pickle
import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
import shutil
import io

def setup_output_dirs(output_dir):
    """Create output directory structure"""
    output_path = Path(output_dir)
    
    # Create main directories
    dirs_to_create = [
        'images/train2014',
        'images/val2014', 
        'text',
        'train',
        'test',
        'DAMSMencoder',
        'npz'
    ]
    
    for dir_name in dirs_to_create:
        (output_path / dir_name).mkdir(parents=True, exist_ok=True)
    
    return output_path

def convert_text_to_caption(text_data, case_name):
    """Convert numerical text data from demo dataset to natural language description"""
    # Parse numerical data
    cleaned_text = text_data.replace(',', ' ').replace(';', ' ')
    values = []
    for x in cleaned_text.split():
        try:
            values.append(float(x))
        except ValueError:
            continue  # Skip non-numeric strings
    
    # Calculate statistics
    max_val = max(values) if values else 0
    min_val = min(values) if values else 0
    avg_val = sum(values) / len(values) if values else 0
    non_zero_count = len([v for v in values if v > 0.001])
    
    # Generate descriptive caption
    if max_val < 0.001:
        caption = f"A clear 3D smoke simulation showing no significant smoke density in case {case_name}"
    elif max_val < 0.1:
        caption = f"A 3D smoke simulation with low density smoke patterns, maximum density {max_val:.3f} in case {case_name}"
    elif max_val < 0.5:
        caption = f"A moderate 3D smoke simulation showing visible smoke distribution with density up to {max_val:.3f} in case {case_name}"
    else:
        caption = f"A dense 3D smoke simulation with high concentration areas, maximum density {max_val:.3f} in case {case_name}"
    
    # Add more descriptive details
    if non_zero_count > len(values) * 0.5:
        caption += " with widespread smoke distribution"
    elif non_zero_count > len(values) * 0.2:
        caption += " with moderate smoke coverage"
    else:
        caption += " with localized smoke patterns"
        
    return caption

def process_parquet_files(data_dir, output_dir, split='train'):
    """Process parquet files and convert to COCO format"""
    demo_path = Path(data_dir)
    output_path = Path(output_dir)
    
    # Determine train or validation set
    if split == 'train':
        parquet_dir = demo_path / 'train'
        image_output_dir = output_path / 'images' / 'train2014'
        split_name = 'train2014'
    else:
        parquet_dir = demo_path / 'validation'
        image_output_dir = output_path / 'images' / 'val2014'
        split_name = 'val2014'
    
    # Collect all parquet files
    parquet_files = list(parquet_dir.glob('**/*.parquet'))
    print(f"Found {len(parquet_files)} parquet files for {split} set")
    
    filenames = []
    captions_data = []
    
    file_counter = 0
    
    for parquet_file in tqdm(parquet_files, desc=f"Processing {split} set"):
        df = pd.read_parquet(parquet_file)
        
        for idx, row in df.iterrows():
            # Generate COCO-style filename
            filename = f"COCO_{split_name}_{file_counter:012d}"
            filenames.append(filename)
            
            # Save image
            if isinstance(row['image'], dict) and 'bytes' in row['image']:
                image_bytes = row['image']['bytes']
                image = Image.open(io.BytesIO(image_bytes))
            else:
                image = row['image']
            
            # Ensure image is RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_path = image_output_dir / f"{filename}.jpg"
            image.save(image_path, 'JPEG', quality=95)
            
            # Generate caption
            caption = convert_text_to_caption(row['text'], row['case'])
            
            # Save caption text file
            caption_path = output_path / 'text' / f"{filename}.txt"
            with open(caption_path, 'w', encoding='utf-8') as f:
                f.write(caption)
            
            # Collect caption data for DAMSM pickle
            captions_data.append({
                'filename': filename,
                'caption': caption,
                'case': row['case']
            })
            
            file_counter += 1
    
    return filenames, captions_data

def create_simple_damsm_data(all_captions, output_dir):
    """Create simplified DAMSM format data"""
    # Build vocabulary
    vocab = set()
    all_words = []
    
    for item in all_captions:
        words = item['caption'].lower().replace(',', '').replace('.', '').split()
        all_words.extend(words)
        vocab.update(words)
    
    # Create vocabulary mapping
    vocab_list = ['<pad>', '<start>', '<end>', '<unk>'] + sorted(list(vocab))
    word_to_ix = {word: i for i, word in enumerate(vocab_list)}
    ix_to_word = {i: word for word, i in word_to_ix.items()}
    
    # Simplified captions data structure
    captions_encoded = []
    for item in all_captions:
        words = item['caption'].lower().replace(',', '').replace('.', '').split()
        encoded = [word_to_ix.get(word, word_to_ix['<unk>']) for word in words]
        captions_encoded.append(encoded)
    
    # Create simplified DAMSM format data
    damsm_data = [
        captions_encoded,  # Encoded captions
        [],  # Empty image features (DF-GAN will recalculate during training)
        word_to_ix,       # Word to index mapping
        ix_to_word        # Index to word mapping
    ]
    
    # Save DAMSM data
    damsm_path = Path(output_dir) / 'captions_DAMSM.pickle'
    with open(damsm_path, 'wb') as f:
        pickle.dump(damsm_data, f)
    
    print(f"Created vocabulary with {len(vocab_list)} words")
    print(f"Saved DAMSM data to: {damsm_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert demo dataset to DF-GAN COCO format')
    parser.add_argument('--data_dir', type=str, 
                       default='/home/dszh/workspace/tmp-smoke/Python/data/demo',
                       help='Path to demo dataset')
    parser.add_argument('--output_dir', type=str,
                       default='/home/dszh/workspace/tmp-smoke/Python/examples/benchmark/references/DF-GAN/data/demo_coco',
                       help='Output path for COCO format dataset')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing output directory')
    
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    
    # Check output directory
    if output_path.exists() and not args.overwrite:
        print(f"Output directory {output_path} already exists. Use --overwrite to overwrite.")
        return
    
    if output_path.exists() and args.overwrite:
        print(f"Deleting existing output directory: {output_path}")
        shutil.rmtree(output_path)
    
    # Setup output directory
    print("Creating output directory structure...")
    setup_output_dirs(args.output_dir)
    
    # Process training set
    print("Processing training set...")
    train_filenames, train_captions = process_parquet_files(
        args.data_dir, args.output_dir, 'train'
    )
    
    # Process validation set
    print("Processing validation set...")
    val_filenames, val_captions = process_parquet_files(
        args.data_dir, args.output_dir, 'validation'
    )
    
    # Save filenames.pickle
    train_pickle_path = output_path / 'train' / 'filenames.pickle'
    with open(train_pickle_path, 'wb') as f:
        pickle.dump(train_filenames, f)
    print(f"Saved training filenames to: {train_pickle_path} ({len(train_filenames)} files)")
    
    test_pickle_path = output_path / 'test' / 'filenames.pickle'
    with open(test_pickle_path, 'wb') as f:
        pickle.dump(val_filenames, f)
    print(f"Saved test filenames to: {test_pickle_path} ({len(val_filenames)} files)")
    
    # Create DAMSM data
    print("Creating DAMSM format data...")
    all_captions = train_captions + val_captions
    create_simple_damsm_data(all_captions, args.output_dir)
    
    # Copy necessary files (if exist)
    original_coco_dir = Path('/home/dszh/workspace/tmp-smoke/Python/examples/benchmark/references/DF-GAN/data/coco')
    if original_coco_dir.exists():
        # Copy DAMSMencoder files
        damsm_encoder_src = original_coco_dir / 'DAMSMencoder'
        damsm_encoder_dst = output_path / 'DAMSMencoder'
        if damsm_encoder_src.exists():
            for file_path in damsm_encoder_src.glob('*.pth'):
                shutil.copy2(file_path, damsm_encoder_dst)
                print(f"Copied DAMSM encoder: {file_path.name}")
        
        # Copy npz files
        npz_src = original_coco_dir / 'npz'
        npz_dst = output_path / 'npz'
        if npz_src.exists():
            for file_path in npz_src.glob('*.npz'):
                shutil.copy2(file_path, npz_dst)
                print(f"Copied NPZ file: {file_path.name}")
    
    print(f"\n✅ Conversion completed!")
    print(f"Output directory: {output_path}")
    print(f"Training set: {len(train_filenames)} samples")
    print(f"Test set: {len(val_filenames)} samples")
    print(f"Total: {len(train_filenames) + len(val_filenames)} samples")


if __name__ == '__main__':
    main()
