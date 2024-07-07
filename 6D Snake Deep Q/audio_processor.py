
import librosa
import numpy as np

def get_frequency_data(file_path, n_fft=2048, hop_length=44):
    """
    Load an audio file and compute its Short-time Fourier Transform (STFT).

    Parameters:
    file_path (str): Path to the audio file.
    n_fft (int): Number of FFT components (default: 2048).
    hop_length (int): Number of samples between successive frames (default: 512).

    Returns:
    tuple: (stft_mag, time_axis, freq_axis, y, sr)
        stft_mag (np.ndarray): Magnitude of the STFT.
        time_axis (np.ndarray): Time values for each frame.
        freq_axis (np.ndarray): Frequency values for each bin.
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate of y.
    """
    # Load the audio file
    y, sr = librosa.load(file_path)

    # Compute the Short-time Fourier Transform (STFT)
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

    # Convert to magnitude spectrogram
    stft_mag = np.abs(stft)

    # Create time and frequency axes
    time_axis = librosa.times_like(stft, sr=sr, hop_length=hop_length)
    freq_axis = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    return stft_mag, time_axis, freq_axis, y, sr
