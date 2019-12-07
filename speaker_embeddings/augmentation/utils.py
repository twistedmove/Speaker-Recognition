import librosa
import random
import numpy as np
import scipy.io.wavfile as wavfile


def mel_spectrogram(data: np.ndarray, n_fft: int, hop_length: int, win_length: int, n_mels: int) -> np.ndarray:
    # mel-scaled spectrogram
    spectrogram = librosa.feature.melspectrogram(data, n_fft=n_fft, hop_length=hop_length,
                                                 win_length=win_length, n_mels=n_mels)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram


def linear_spectrogram(data: np.ndarray, n_fft: int, hop_length: int, win_length: int) -> np.ndarray:
    # linear spectrogram
    spectrogram = librosa.stft(data, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    return spectrogram


def mfcc(mel_spectrogram: np.ndarray, n_mfcc: int = 20) -> np.ndarray:
    # Mel-frequency cepstral coefficients
    return librosa.feature.mfcc(S=mel_spectrogram, n_mfcc=n_mfcc)


def load_audio(path: str, sample_rate: int = 16000, ignore_zero: bool = False) -> np.ndarray:

    assert path.split('.')[-1] == 'wav'

    # librosa slower then the wave to load
    #audio, sr_ret = librosa.load(path, sr=sample_rate)
    sr_ret, audio = wavfile.read(path)

    assert sr_ret == sample_rate

    audio = np.array(audio, dtype=np.float32)
    audio /= 2 ** 15

    if ignore_zero:
        audio = audio[audio != 0]

    return audio


def crop(data: np.ndarray, spectrogram_length: int) -> np.ndarray:

    freq, time = data.shape
    randtime = np.random.randint(0, time - spectrogram_length)
    data = data[:, randtime:randtime + spectrogram_length]
    return data


def magnitude(spectrogram: np.ndarray) -> np.ndarray:
    magnitude, _ = librosa.magphase(spectrogram)  # magnitude
    return magnitude


def normalize(magnitude: np.ndarray, eps=1e-5) -> np.ndarray:

    # preprocessing, subtract mean, divided by time-wise var
    mu = np.mean(magnitude, 0, keepdims=True)
    std = np.std(magnitude, 0, keepdims=True)
    return (magnitude - mu) / (std + eps)


def to_tensor(x, **kwargs):
    """
    Convert spectrogram.
    Args:
        x:
        **kwargs:
    """
    return np.expand_dims(x, axis=0).astype('float32')


def spec_augment(spec: np.ndarray,
                 num_mask=2,
                 freq_masking=0.15,
                 time_masking=0.20,
                 value=0):
    spec = spec.copy()
    num_mask = random.randint(1, num_mask)
    for i in range(num_mask):
        all_freqs_num, all_frames_num = spec.shape
        freq_percentage = random.uniform(0.0, freq_masking)

        num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
        f0 = int(f0)
        spec[f0:f0 + num_freqs_to_mask, :] = value

        time_percentage = random.uniform(0.0, time_masking)

        num_frames_to_mask = int(time_percentage * all_frames_num)
        t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
        t0 = int(t0)
        spec[:, t0:t0 + num_frames_to_mask] = value
    return spec
