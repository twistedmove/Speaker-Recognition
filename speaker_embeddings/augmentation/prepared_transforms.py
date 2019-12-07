from .composition import Compose, OneOf
from .transforms import *


def get_training_augmentation(spectrogram_length: int = 250, **kwargs):

    augmenter = Compose([
        Append(p=0.3),
        Inverse(p=0.2),
        ToLinearSpectrogram(**kwargs),
        ToMagnitude(),
        RandomCrop(spectrogram_length=spectrogram_length),
        NormalizeSpectrogram(),
        SpecAugment(p=0.4),
        ToTensor(),
    ], p=1.0)

    return augmenter


def get_valid_augmentation(spectrogram_length: int = 250, **kwargs):

    augmenter = Compose([
        ToLinearSpectrogram(**kwargs),
        ToMagnitude(),
        RandomCrop(spectrogram_length=spectrogram_length),
        NormalizeSpectrogram(),
        ToTensor(),
    ])

    return augmenter


def get_eval_augmentation(n_fft: int = 512, hop_length: int = 160,
                           win_length: int = 400, n_mels: int = 128):

    augmenter = Compose([
        ToMelSpectrogram(n_fft, hop_length, win_length, n_mels),
        Transpose(),
        ToMagnitude(),
        Transpose(),
        NormalizeSpectrogram(),
        ToTensor(),
    ])

    return augmenter
