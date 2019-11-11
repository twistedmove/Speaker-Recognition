import pandas as pd
import numpy as np
from collections import OrderedDict
from .augmentation import load_audio
from typing import Dict
from catalyst.data.augmentor import Augmentor
from torchvision import transforms
from catalyst.dl import utils as catalyst_utils
from catalyst.data.reader import ScalarReader, ReaderCompose, ReaderSpec


class AudioReader(ReaderSpec):
    """
    Audio reader abstraction. Reads audio from a `csv` dataset.
    """
    def __init__(
        self,
        input_key: str,
        output_key: str,
    ):
        """
        Args:
            input_key (str): key to use from annotation dict
            output_key (str): key to use to store the result
        """
        super().__init__(input_key, output_key)

    def __call__(self, row):
        """Reads a row from your annotations dict with filename and
        transfer it to an audio

        Args:
            row: elem in your dataset.

        Returns:
            np.ndarray: Audio
        """
        audio_name = str(row[self.input_key])
        audio = load_audio(audio_name, 16000)
        return {self.output_key: audio.copy()}


def _prepare(data_file, root_folder):

    # example of row id00012/s7MiVWybRhg/00150.wav 0

    df = pd.read_csv(data_file, sep='\s', engine='python', names=["filepath", "label"])

    df['label'] = df['label'].astype(np.uint8)

    df['filepath'] = df['filepath'].apply(lambda x: "{}/{}".format(root_folder, x))

    dataset = df.to_dict('records')

    return dataset


def create_dataloders(train_file: str,
                      valid_file: str,
                      root_folder: str,
                      meta_info_file: str,
                      num_classes: int,
                      one_hot_encoding: bool,
                      bs: int,
                      num_workers: int,
                      augmenters: Dict = None,
                      ):

    train_data = _prepare(train_file, root_folder)
    valid_data = _prepare(valid_file, root_folder)

    train_augmenter = augmenters['train']
    valid_augmenter = augmenters['valid']

    train_transforms_fn = transforms.Compose([
        Augmentor(
            dict_key="features",
            augment_fn=lambda x: train_augmenter(samples=x, sample_rate=16000)
        )
    ])

    # Similarly for the validation part of the dataset.
    # we only perform squaring, normalization and ToTensor
    valid_transforms_fn = transforms.Compose([
        Augmentor(
            dict_key="features",
            augment_fn=lambda x: valid_augmenter(samples=x, sample_rate=16000)
        )
    ])

    compose = [

        AudioReader(
            input_key="filepath",
            output_key="features",
        ),
        ScalarReader(
            input_key="label",
            output_key="targets",
            default_value=-1,
            dtype=np.int64
        ),
    ]

    if one_hot_encoding:
        compose.append(ScalarReader(
            input_key="label",
            output_key="targets_one_hot",
            default_value=-1,
            dtype=np.int64,
            one_hot_classes=num_classes,
        ))

    open_fn = ReaderCompose(compose)

    train_loader = catalyst_utils.get_loader(
        train_data,
        open_fn=open_fn,
        dict_transform=train_transforms_fn,
        batch_size=bs,
        num_workers=num_workers,
        shuffle=True,  # shuffle data only if Sampler is not specified (PyTorch requirement)
    )

    valid_loader = catalyst_utils.get_loader(
        valid_data,
        open_fn=open_fn,
        dict_transform=valid_transforms_fn,
        batch_size=bs,
        num_workers=1,
        shuffle=False,
    )

    loaders = OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = valid_loader

    return loaders
