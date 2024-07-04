import librosa
import numpy as np

def extract_timeseries_data(filename):
    # Load the audio file
    y, sr = librosa.load(filename)
    
    # Calculate the time series of parameters
    amplitude = np.abs(y)
    rms = librosa.feature.rms(y=y)[0]
    zero_crossings = librosa.feature.zero_crossing_rate(y)[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Create a time axis for the amplitude
    t_amplitude = np.arange(len(amplitude)) / sr
    
    # Create a time axis for the features
    t_features = librosa.frames_to_time(np.arange(rms.shape[0]), sr=sr)
    
    # Print the parameters with their corresponding time values
    print("Time (Amplitude):", t_amplitude)
    print("Amplitude:", amplitude)
    print("Time (RMS Energy):", t_features)
    print("RMS Energy:", rms)
    print("Time (Zero Crossing Rate):", t_features)
    print("Zero Crossing Rate:", zero_crossings)
    print("Time (Spectral Centroid):", t_features)
    print("Spectral Centroid:", spectral_centroid)
    print("Time (Spectral Bandwidth):", t_features)
    print("Spectral Bandwidth:", spectral_bandwidth)
    print("Time (Spectral Contrast):", t_features)
    print("Spectral Contrast:", spectral_contrast)
    print("Time (MFCCs):", t_features)
    print("MFCCs:", mfccs)

    # Return the extracted parameters with their corresponding time values
    return {
        'time_amplitude': t_amplitude,
        'amplitude': amplitude,
        'time_features': t_features,
        'rms': rms,
        'zero_crossings': zero_crossings,
        'spectral_centroid': spectral_centroid,
        'spectral_bandwidth': spectral_bandwidth,
        'spectral_contrast': spectral_contrast,
        'mfccs': mfccs
    }

# Example usage
filename = 'music/audio.mp4'
data = extract_timeseries_data(filename)
print(data)
