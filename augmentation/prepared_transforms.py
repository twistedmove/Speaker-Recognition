from .composition import Compose, OneOf
from .transforms import *


def get_training_augmentation():

    augmenter = Compose([
        OneOf([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
            Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
        ], p=0.3),
        ToSpectrogram(512, 160, 400, 0),
        Transpose(),
        ToMagnitude(),
        Transpose(),
        RandomCrop(spectrogram_length=250, p=1.0),
        NormalizeSpectrogram(),
        ToTensor(),
    ], p=1.0)

    return augmenter


def get_valid_augmentation():

    augmenter = Compose([
        ToSpectrogram(512, 160, 400, 0),
        Transpose(),
        ToMagnitude(),
        Transpose(),
        RandomCrop(spectrogram_length=250),
        NormalizeSpectrogram(),
        ToTensor(),
    ])

    return augmenter


if __name__ == '__main__':

    from .utils import load_audio

    augmenter = get_training_augmentation()

    audio = load_audio('/home/user/PycharmProjects/test0.wav', 16000)

    transformed = augmenter(audio)
