import pandas as pd
import numpy as np
from collections import OrderedDict
from augmentation import load_audio
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
        root: str,
    ):
        """
        Args:
            input_key (str): key to use from annotation dict
            output_key (str): key to use to store the result
        """
        super().__init__(input_key, output_key)
        self.root = root

    def __call__(self, row):
        """Reads a row from your annotations dict with filename and
        transfer it to an image

        Args:
            row: elem in your dataset.

        Returns:
            np.ndarray: Image
        """
        audio_name = str(row[self.input_key])
        audio = load_audio(audio_name, 16000)

        result = {self.output_key: audio}
        return result


def _prepare(meta_info_file, data_file, root_folder):

    meta = pd.read_csv(meta_info_file)

    meta_dict = pd.Series(meta['Gender '].values, index=meta['VoxCeleb2 ID '].apply(lambda x: x.strip())).to_dict()

    df = pd.read_csv(data_file, sep='\s', engine='python', names=["filepath", "label"])

    df['label'] = df['label'].astype(np.int64)

    df['id'] = df['filepath'].apply(lambda x: x.split('/')[0])

    df['gender'] = df['id'].map(meta_dict).apply(lambda x: 0 if x == 'm' else 1)

    df['filepath'] = df['filepath'].apply(lambda x: "{}/{}".format(root_folder, x))

    dataset = df.to_dict('records')

    return dataset


def create_dataloders(train_file: str,
                      valid_file: str,
                      root_folder: str,
                      meta_info_file: str,
                      num_classes: int,
                      bs: int,
                      num_workers: int,
                      augmenters: Dict = None,
                      ):

    train_data = _prepare(meta_info_file, train_file, root_folder)
    valid_data = _prepare(meta_info_file, valid_file, root_folder)

    train_transforms_fn = transforms.Compose([
        Augmentor(
            dict_key="features",
            augment_fn=lambda x: augmenters['train'](samples=x, sample_rate=16000)
        )
    ])

    # Similarly for the validation part of the dataset.
    # we only perform squaring, normalization and ToTensor
    valid_transforms_fn = transforms.Compose([
        Augmentor(
            dict_key="features",
            augment_fn=lambda x: augmenters['valid'](samples=x, sample_rate=16000)
        )
    ])

    open_fn = ReaderCompose([
        AudioReader(
            input_key="filepath",
            output_key="features",
            root=root_folder,
        ),
        ScalarReader(
            input_key="label",
            output_key="targets",
            default_value=-1,
            dtype=np.int64
        ),
        ScalarReader(
            input_key="label",
            output_key="targets_one_hot",
            default_value=-1,
            dtype=np.int64,
            one_hot_classes=num_classes,
        )
    ])

    # current time doesn't use sampler
    sampler = None

    train_loader = catalyst_utils.get_loader(
        train_data,
        open_fn=open_fn,
        dict_transform=train_transforms_fn,
        batch_size=bs,
        num_workers=num_workers,
        shuffle=sampler is None,  # shuffle data only if Sampler is not specified (PyTorch requirement)
        sampler=sampler
    )

    valid_loader = catalyst_utils.get_loader(
        valid_data,
        open_fn=open_fn,
        dict_transform=valid_transforms_fn,
        batch_size=bs,
        num_workers=num_workers,
        shuffle=False,
        sampler=None
    )

    loaders = OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = valid_loader

    return loaders
