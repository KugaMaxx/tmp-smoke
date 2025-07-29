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
_CITATION = "TODO"

_FEATURES = datasets.Features(
    {
        "image": datasets.Image(),
        "text": datasets.Value("string"),
    },
)

_DEFAULT_CONFIG = datasets.BuilderConfig(name="default", version=_VERSION)


class MyDataset(datasets.GeneratorBasedBuilder):
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
                    "metadata_path": base_path / "train" / "prompt.jsonl",
                    "images_dir": base_path,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "metadata_path": base_path / "validation" / "prompt.jsonl",
                    "images_dir": base_path,
                },
            ),
        ]

    def _generate_examples(self, metadata_path, images_dir):
        metadata = pd.read_json(metadata_path, lines=True)

        for _, row in metadata.iterrows():
            text = row["text"]

            image_path = row["image"]
            image_path = os.path.join(images_dir, image_path)
            image = open(image_path, "rb").read()

            yield row["image"], {
                "text": text,
                "image": {
                    "path": image_path,
                    "bytes": image,
                }
            }
