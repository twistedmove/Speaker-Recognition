import abc
import librosa
import random
import numpy as np
from scipy.signal import butter, lfilter
from .utils import crop, to_tensor, linear_spectrogram, mel_spectrogram, normalize, spectrogram2magnitude


class BasicTransform(metaclass=abc.ABCMeta):
    def __init__(self, p=0.5):
        assert 0 <= p <= 1
        self.p = p

    def __call__(self, samples: np.ndarray, **kwargs):
        if random.random() < self.p:
            return self.apply(samples, **kwargs)
        return samples

    @abc.abstractmethod
    def apply(self, samples: np.ndarray, **kwargs):
        pass


class AddGaussianNoise(BasicTransform):
    """Add gaussian noise to the samples"""

    def __init__(self, min_amplitude=0.001, max_amplitude=0.015, p=0.5):
        super().__init__(p)
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude

    def apply(self, samples, **kwargs):
        samples = samples.copy()
        noise = np.random.randn(len(samples)).astype(np.float32)
        amplitude = random.uniform(self.min_amplitude, self.max_amplitude)
        samples = samples + amplitude * noise
        return samples


class PitchShift(BasicTransform):
    """Pitch shift the sound up or down without changing the tempo"""

    def __init__(self, min_semitones=-4, max_semitones=4, p=0.5):
        super().__init__(p)
        assert min_semitones >= -12
        assert max_semitones <= 12
        assert min_semitones <= max_semitones
        self.min_semitones = min_semitones
        self.max_semitones = max_semitones

    def apply(self, samples, **kwargs):
        samples = samples.copy()
        sr = kwargs['sample_rate']
        num_semitones = random.uniform(self.min_semitones, self.max_semitones)
        pitch_shifted_samples = librosa.effects.pitch_shift(
            samples, sr=sr, n_steps=num_semitones
        )
        return pitch_shifted_samples


class Shift(BasicTransform):
    """
    Shift the samples forwards or backwards. Samples that roll beyond the first or last position
    are re-introduced at the last or first.
    """

    def __init__(self, min_fraction=-0.5, max_fraction=0.5, p=0.5):
        """
        :param min_fraction: float, fraction of total sound length
        :param max_fraction: float, fraction of total sound length
        :param p:
        """
        super().__init__(p)
        assert min_fraction >= -1
        assert max_fraction <= 1
        self.min_fraction = min_fraction
        self.max_fraction = max_fraction

    def apply(self, samples, **kwargs):
        samples = samples.copy()
        num_places_to_shift = int(
            round(random.uniform(self.min_fraction, self.max_fraction) * len(samples))
        )
        shifted_samples = np.roll(samples, num_places_to_shift)
        return shifted_samples


class TimeStretch(BasicTransform):
    """Time stretch the signal without changing the pitch"""

    def __init__(self, min_rate=0.8, max_rate=1.25, leave_length_unchanged=True, p=0.5):
        super().__init__(p)
        assert min_rate > 0.1
        assert max_rate < 10
        assert min_rate <= max_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.leave_length_unchanged = leave_length_unchanged

    def apply(self, samples, **kwargs):
        """
        If `rate > 1`, then the signal is sped up.
        If `rate < 1`, then the signal is slowed down.
        """
        samples = samples.copy()
        rate = random.uniform(self.min_rate, self.max_rate)
        time_stretched_samples = librosa.effects.time_stretch(samples, rate)
        if self.leave_length_unchanged:
            # Apply zero padding if the time stretched audio is not long enough to fill the
            # whole space, or crop the time stretched audio if it ended up too long.
            padded_samples = np.zeros(shape=samples.shape, dtype=samples.dtype)
            window = time_stretched_samples[: samples.shape[0]]
            actual_window_length = len(window)  # may be smaller than samples.shape[0]
            padded_samples[:actual_window_length] = window
            time_stretched_samples = padded_samples
        return time_stretched_samples


class FrequencyMask(BasicTransform):
    """Mask some frequency band on the spectrogram. Inspired by https://arxiv.org/pdf/1904.08779.pdf """

    def __init__(self, min_frequency_band=0.0, max_frequency_band=0.5, p=0.5):
        """
        :param min_frequency_band: Minimum bandwidth, float
        :param max_frequency_band: Maximum bandwidth, float
        :param p:
        """
        super().__init__(p)
        self.min_frequency_band = min_frequency_band
        self.max_frequency_band = max_frequency_band

    def __butter_bandstop(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype="bandstop")
        return b, a

    def __butter_bandstop_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.__butter_bandstop(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data).astype(np.float32)
        return y

    def apply(self, samples, **kwargs):
        sample_rate = kwargs['sample_rate']
        band_width = random.randint(
            self.min_frequency_band * sample_rate // 2,
            self.max_frequency_band * sample_rate // 2,
        )
        freq_start = random.randint(16, sample_rate / 2 - band_width)
        samples = self.__butter_bandstop_filter(
            samples.copy(), freq_start, freq_start + band_width, sample_rate, order=6
        )
        return samples


class TimeMask(BasicTransform):
    """Mask some time band on the spectrogram. Inspired by https://arxiv.org/pdf/1904.08779.pdf """

    def __init__(self, min_band_part=0.0, max_band_part=0.5, p=0.5):
        """
        :param min_band_part: Minimum length of the silent part as a fraction of the
            total sound length. Float.
        :param max_band_part: Maximum length of the silent part as a fraction of the
            total sound length. Float.
        :param p:
        """
        super().__init__(p)
        self.min_band_part = min_band_part
        self.max_band_part = max_band_part

    def apply(self, samples, **kwargs):
        new_samples = samples.copy()
        _t = random.randint(
            int(new_samples.shape[0] * self.min_band_part),
            int(new_samples.shape[0] * self.max_band_part),
        )
        _t0 = random.randint(0, new_samples.shape[0] - _t)
        new_samples[_t0 : _t0 + _t] = 0
        return new_samples


class RandomCrop(BasicTransform):
    """Crop a random part of the input.
    """

    def __init__(self, spectrogram_length: int = 250, p: float = 1.0):
        super().__init__(p)
        self.spectrogram_length = spectrogram_length
        self.func = crop

    def apply(self, samples, **kwargs):
        return self.func(samples.copy(), self.spectrogram_length)


class Append(BasicTransform):
    """Double the input.
    """

    def __init__(self, reverse=True, p: float = 1.0):
        super().__init__(p)
        self.direction = -1 if reverse else 1

    def apply(self, samples, **kwargs):
        return np.append(samples, samples[::self.direction]).copy()


class ToTensor(BasicTransform):
    """Convert numpy.ndarray to tensor.
    """

    def __init__(self, p: float = 1.0):
        super().__init__(p)
        self.func = to_tensor

    def apply(self, samples, **kwargs):
        return self.func(samples).copy()


class Transpose(BasicTransform):
    """Transpose numpy.ndarray
    """

    def __init__(self, p=1.0):
        super().__init__(p)
        self.func = np.transpose

    def apply(self, samples, **kwargs):
        return self.func(samples).copy()


class ToSpectrogram(BasicTransform):
    """Convert wav audio to spectrogram.
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int, kind: int = 0, p=1.0):
        super().__init__(p)
        self.func = linear_spectrogram if kind == 0 else mel_spectrogram
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.kind = kind

    def apply(self, samples, **kwargs):
        return self.func(samples, self.n_fft, self.hop_length, self.win_length).copy()


class NormalizeSpectrogram(BasicTransform):
    """Subtract mean and divide on std
    """

    def __init__(self, p=1.0):
        super().__init__(p)
        self.func = normalize

    def apply(self, samples, **kwargs):
        return self.func(samples).copy()


class ToMagnitude(BasicTransform):
    """
    """

    def __init__(self, p=1.0):
        super().__init__(p)
        self.func = spectrogram2magnitude

    def apply(self, samples, **kwargs):
        return self.func(samples).copy()
