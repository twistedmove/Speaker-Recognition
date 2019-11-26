from .composition import Compose, OneOf
from .transforms import *


def get_training_augmentation(n_fft: int = 512, hop_length: int = 160,
                              win_length: int = 400, n_mels: int = 128,
                              spectrogram_length: int = 250):

    augmenter = Compose([
        OneOf([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
        ], p=0.3),
        ToMelSpectrogram(n_fft, hop_length, win_length, n_mels),
        RandomCrop(spectrogram_length=spectrogram_length, p=1.0),
        NormalizeSpectrogram(),
        SpecAugment(),
        ToTensor(),
    ], p=1.0)

    return augmenter


def get_valid_augmentation(n_fft: int = 512, hop_length: int = 160,
                           win_length: int = 400, n_mels: int = 128,
                           spectrogram_length: int = 250):

    augmenter = Compose([
        ToMelSpectrogram(n_fft, hop_length, win_length, n_mels),
        RandomCrop(spectrogram_length=spectrogram_length, p=1.0),
        NormalizeSpectrogram(),
        ToTensor(),
    ])

    return augmenter


def get_eval_augmentation(n_fft: int = 512, hop_length: int = 160,
                              win_length: int = 400, kind: int = 0):

    augmenter = Compose([
        ToSpectrogram(n_fft, hop_length, win_length, kind),
        Transpose(),
        ToMagnitude(),
        Transpose(),
        NormalizeSpectrogram(),
        ToTensor(),
    ])

    return augmenter


if __name__ == '__main__':

    from .utils import load_audio

    augmenter = get_training_augmentation()

    for i in range(0, 1000):

        audio = load_audio('/data_ssd/VoxCeleb2/vox2_dev_dataset/dev/aac/id00016/3Ikj1W8iEyE/00012.wav', 16000)

        transformed = augmenter(audio)
