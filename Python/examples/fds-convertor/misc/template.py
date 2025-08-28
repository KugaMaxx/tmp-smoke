#! /usr/bin/env python3
# This module defines a custom HuggingFace `datasets.GeneratorBasedBuilder` for 
# loading an AIoT-based smoke reconstruction dataset. You may find references at
#   https://medium.com/@dangattringer/step-by-step-guide-loading-a-huggingface-controlnet-dataset-from-a-local-path-9b00b81f06b7

import os
import datasets
import pandas as pd
from pathlib import Path


_VERSION = datasets.Version("1.0.0")

_DESCRIPTION = "Dataset for AIoT-based smoke reconstruction."
_HOMEPAGE = "https://github.com/KugaMaxx"
_LICENSE = "MIT LICENSE"
_CITATION = "NONE"

_FEATURES = datasets.Features(
    {
        "image": datasets.Image(),
        "text": datasets.Value("string"),
    },
)

_DEFAULT_CONFIG = datasets.BuilderConfig(name="default", version=_VERSION)


class CustomDataset(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [_DEFAULT_CONFIG]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=_FEATURES,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        base_path = Path(dl_manager._base_path).resolve()
        dl_manager.download_and_extract
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "metadata_files": list((base_path / "train").glob("*/*.jsonl")),
                    "image_dirs": list((base_path / "train").glob("*")),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "metadata_files": list((base_path / "validation").glob("*/*.jsonl")),
                    "image_dirs": list((base_path / "validation").glob("*")),
                },
            ),
        ]

    def _generate_examples(self, metadata_files, image_dirs):
        for metadata_file, image_dir in zip(metadata_files, image_dirs):
            # Read the metadata file
            metadata = pd.read_json(metadata_file, lines=True)

            for _, row in metadata.iterrows():
                text = row["text"]

                image_path = row["image"]
                image_path = os.path.join(image_dir, image_path)
                image = open(image_path, "rb").read()

                yield row["image"], {
                    "text": text,
                    "image": {
                        "path": image_path,
                        "bytes": image,
                    }
                }
