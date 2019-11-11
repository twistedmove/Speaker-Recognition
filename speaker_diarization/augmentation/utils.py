import librosa
import random
import numpy as np
import scipy.io.wavfile as wavfile


def mel_spectrogram(data: np.ndarray, n_fft: int, hop_length: int, win_length: int) -> np.ndarray:
    # mel-scaled spectrogram
    spectrogram = librosa.feature.melspectrogram(data, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    return spectrogram.T


def linear_spectrogram(data: np.ndarray, n_fft: int, hop_length: int, win_length: int) -> np.ndarray:
    # linear spectrogram
    spectrogram = librosa.stft(data, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    return spectrogram


def mfcc(transpose_mel_spectrogram: np.ndarray, n_mfcc: int = 20) -> np.ndarray:
    # Mel-frequency cepstral coefficients
    return librosa.feature.mfcc(S=transpose_mel_spectrogram.T, n_mfcc=n_mfcc)


def load_audio(path: str, sample_rate: int = 16000, ignore_zero: bool = False) -> np.ndarray:

    assert path.split('.')[-1] == 'wav'

    # librosa slower then the wave to load
    #audio, sr_ret = librosa.load(path, sr=sample_rate)
    sr_ret, audio = wavfile.read(path)
    audio = np.array(audio, dtype=np.float32)
    audio /= 2 ** 15

    if ignore_zero:
        audio = audio[audio != 0]

    assert sr_ret == sample_rate

    return audio


def crop(data: np.ndarray, spectrogram_length: int) -> np.ndarray:

    freq, time = data.shape
    randtime = np.random.randint(0, time - spectrogram_length)
    data = data[:, randtime:randtime + spectrogram_length]
    return data


def spectrogram2magnitude(spectrogram: np.ndarray) -> np.ndarray:
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


def spec_augment(spec: np.ndarray, num_mask=2,
                 freq_masking_max_percentage=0.3, time_masking_max_percentage=0.3):
    spec = spec.copy()
    for i in range(num_mask):
        all_frames_num, all_freqs_num = spec.shape
        freq_percentage = random.uniform(0.0, freq_masking_max_percentage)

        num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
        f0 = int(f0)
        spec[:, f0:f0 + num_freqs_to_mask] = 0

        time_percentage = random.uniform(0.0, time_masking_max_percentage)

        num_frames_to_mask = int(time_percentage * all_frames_num)
        t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
        t0 = int(t0)
        spec[t0:t0 + num_frames_to_mask, :] = 0

    return spec
