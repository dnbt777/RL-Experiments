
import pygame
import math
import time
import librosa
import numpy as np
from math import *
from audio_processor import get_frequency_data

song_path = "music/usandthem.ogg"
play_song = True

# Load audio and extract features
y, sr = librosa.load(song_path)

class ParamSpace:
    def __init__(self):
        self.sim_start = time.time()
        self.time = 0
        
        # Extract audio features
        print('loading amplitude')
        self.amplitude = np.abs(y)
        self.amplitude_avg = np.nanmean(self.amplitude)
        print('loading rms')
        self.rms = librosa.feature.rms(y=y)[0]
        print('loading zero_crossings')
        self.zero_crossings = librosa.feature.zero_crossing_rate(y)[0]
        print('loading spectral_centroid')
        self.spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        print('loading spectral_bandwidth')
        self.spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        print('loading chroma')
        self.chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        print('loading formants')
        self.formants = librosa.lpc(y, order=8)
        print(self.formants)
        print('loading mfccs')
        self.mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Create time arrays
        print('loading t_amplitude')
        self.t_amplitude = np.arange(len(self.amplitude)) / sr
        print('loading t_features')
        self.t_features = librosa.frames_to_time(np.arange(len(self.rms)), sr=sr)

        # Load frequency data
        print('loading frequency data')
        self.stft_mag, self.time_axis, self.freq_axis, _, _ = get_frequency_data(song_path)


    def get_mfccs(self, t):
        idx = np.searchsorted(self.t_features, t, side="right") - 1
        return self.mfccs[:, idx]

    def get_freq_data(self, t):
        idx = np.searchsorted(self.time_axis, t, side="right") - 1
        print(idx)
        print()
        return self.stft_mag[:, idx]

    def get_avg_freq_data(self, t):
        freq_data = self.get_freq_data(t)
        return np.mean(freq_data)

    def update(self):
        self.time = time.time() - self.sim_start

    def get_time(self):
        return self.time

    def interpolate(self, t, times, values):
        if t <= times[0]:
            return values[0]
        if t >= times[-1]:
            return values[-1]
        idx = np.searchsorted(times, t, side="right") - 1
        t1, t2 = times[idx], times[idx + 1]
        v1, v2 = values[idx], values[idx + 1]
        return v1 + (v2 - v1) * (t - t1) / (t2 - t1)

    def get_amplitude(self, t):
        amp = self.interpolate(t, self.t_amplitude, self.amplitude)
        if np.isnan(amp):
            amp = self.amplitude_avg
        return amp

    def get_rms(self, t):
        return self.interpolate(t, self.t_features, self.rms)

    def get_zero_crossings(self, t):
        return self.interpolate(t, self.t_features, self.zero_crossings)

    def get_spectral_centroid(self, t):
        return self.interpolate(t, self.t_features, self.spectral_centroid)

    def get_spectral_bandwidth(self, t):
        return self.interpolate(t, self.t_features, self.spectral_bandwidth)

    def get_chroma(self, t):
        idx = np.searchsorted(self.t_features, t, side="right") - 1
        return self.chroma[:, idx]

    def get_formants(self, t):
        idx = np.searchsorted(self.t_amplitude, t, side="right") - 1
        if idx < 0 or idx >= len(self.formants):
            print(f"Warning: Invalid index {idx} for formants at time {t}")
            return np.zeros(8)  # Return an array of zeros with the same shape as formants
        return self.formants[idx]


MODEL_SAVE_INTERVAL = 100000 # steps

# DQN Hyperparameters
BATCH_SIZE = 256
GAMMA = 0.5
EPSILON = 0
EPSILON_MIN = 0.000000
EPSILON_STEPS = 1_000_000
EPSILON_DECAY = (EPSILON - EPSILON_MIN)/EPSILON_STEPS
LR = 1e-4
ALPHA = LR

TAU = 0.005
MEMORY_SIZE = 2048

RENDER_EVERY = 10000
PRINT_EVERY = 10

FC_LAYERS = [
    [512],  # auto input_size, 512
    [512, 512],
    [512, 256],
    [256, 256],
    [256, 128],
    [128, 64],
    [64, 64],
    [64, 64],
    [64],  # auto 16, n_actions size
]

FPS = 120
FRAMES_PER_SNAKE_MOVEMENT = FPS//60

# Screen dimensions
WIDTH, HEIGHT = 900, 900
GRID_SIZE = 10
NUM_GRIDS_W = 4
NUM_GRIDS_V = 4
NUM_GRIDS_U = 4

APPLE_COUNT = 10

# Define ZOOM and ROT1 as functions
def get_zoom(param_space):
    t = param_space.get_time()
    #amplitude = param_space.get_amplitude(t)
    #rms = param_space.get_rms(t)
    #zoom = max(100*(1+sin(t)/8), 100*(1+sin(t)/8) + 15 * np.tanh(2 * rms) + 20 * np.tanh(3 * amplitude))
    zoom = 90 + 30*param_space.get_avg_freq_data(t)
    print(t, zoom)
    return zoom

# works
def get_rot1(param_space):
    t = param_space.get_time()
    spectral_centroid = param_space.get_spectral_centroid(t)
    return t * 0.15 + 0.05 * np.sin(2 * np.pi * t * 0.1) * spectral_centroid / 5000

def get_rot2(param_space):
    t = param_space.get_time()
    zero_crossings = param_space.get_zero_crossings(t)
    return 0.3 * t

def get_rot_cube1(param_space):
    t = param_space.get_time()
    spectral_bandwidth = param_space.get_spectral_bandwidth(t)
    return 0.8 * t

# works
def get_rot_cube2(param_space):
    t = param_space.get_time()
    chroma = param_space.get_chroma(t)
    chroma_intensity = np.mean(chroma)
    return 50*t + 10*sin(t/10)


def get_skybox_base_color(param_space):
    t = param_space.get_time()
    frequency = 2
    r = int((sin(t * frequency + 0) * 127.5 + 127.5) % 255)
    g = int((sin(t * frequency + 2 * pi / 3) * 127.5 + 127.5) % 255)
    b = int((sin(t * frequency + 4 * pi / 3) * 127.5 + 127.5) % 255)
    d = max(0, 0.05*(1 + sin(t * 2.7 + 0.1 * pi / 3)/2))
    return (r*d, g*d, b*d, d//100)

def get_skybox_size(param_space):
    t = param_space.get_time()
    return 1000 + 10*sin(10*t)

def get_rot1_skybox(param_space):
    t = param_space.get_time()
    spectral_centroid = param_space.get_spectral_centroid(t)
    return t * 0.5 + 0.8 * np.sin(2 * np.pi * t * 0.1) * spectral_centroid / 5000

def get_rot2_skybox(param_space):
    t = param_space.get_time()
    zero_crossings = param_space.get_zero_crossings(t)
    return 10.3 * t



# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Define color configurations as functions
def get_grid_color(param_space):
    t = param_space.get_time()
    a = 20 + 20*sin(t)
    return (77, 77, 77, a)

# Snake stuff

def get_snake_fill_color(param_space):
    return (0, 200, 0, 255)  # Light blue

def get_snake_line_color(param_space):
    return (0, 200, 0, 255)

def get_snake_line_thickness(param_space):
    return 4

# Apple stuff

def get_apple_fill_color(param_space):
    frequency = 4
    t = param_space.get_time()
    r = int((cos(t * frequency + 0) * 127.5 + 127.5) % 255)
    g = int((cos(t * frequency + 2 * pi / 3) * 127.5 + 127.5) % 255)
    b = int((cos(t * frequency + 4 * pi / 3) * 127.5 + 127.5) % 255)
    return (r, g, b, 10)  # Keep red

def get_apple_line_color(param_space):
    frequency = 3
    t = param_space.get_time()
    r = int((cos(t * frequency + 0) * 127.5 + 127.5) % 255)
    g = int((cos(t * frequency + 2 * pi / 3) * 127.5 + 127.5) % 255)
    b = int((cos(t * frequency + 4 * pi / 3) * 127.5 + 127.5) % 255)
    return (g, b, r, 255)

# Rendering configurations
SNAKE_CONNECTION_ALPHA = 1.0

# Define cell dimensions as functions
def get_cell_width(param_space):
    t = param_space.get_time()
    return 1 + sin(t)/2 + cos(1*param_space.get_avg_freq_data(t))

def get_cell_height(param_space):
    t = param_space.get_time()
    return 1 + sin(t)/2 + cos(1*param_space.get_avg_freq_data(t))

def get_cell_length(param_space):
    t = param_space.get_time()
    return 1 + sin(t)/2 + cos(1*param_space.get_avg_freq_data(t))

# New visual parameters
def get_main_grid_axes_color(param_space):
    #red_white_blue = get_red_white_blue_color(param_space)
    return (60, 60, 60, 0)

def get_subgrid_axes_color(param_space):
    frequency = 4
    t = param_space.get_time()
    r = int((cos(t * frequency + 0) * 127.5 + 127.5) % 255)
    g = int((cos(t * frequency + 2 * pi / 3) * 127.5 + 127.5) % 255)
    b = int((cos(t * frequency + 4 * pi / 3) * 127.5 + 127.5) % 255)
    a = int((cos(t * 2.7 + 4 * pi / 3) * 127.5 + 127.5) % 255)
    return (r, g, b, a)

def get_main_grid_line_thickness(param_space):
    return 0.1

def get_subgrid_line_thickness(param_space):
    t = param_space.get_time()
    return 0.1 + 0.3*(1+sin(t)) # original: 0.1


EPISODES = 1_000_000_000

# 6D movement directions (absolute)
BASIS_DIRECTIONS = [
    (1, 0, 0, 0, 0, 0),   # x+
    (-1, 0, 0, 0, 0, 0),  # x-
    (0, 1, 0, 0, 0, 0),   # y+
    (0, -1, 0, 0, 0, 0),  # y-
    (0, 0, 1, 0, 0, 0),   # z+
    (0, 0, -1, 0, 0, 0),  # z-
    (0, 0, 0, 1, 0, 0),   # w+
    (0, 0, 0, -1, 0, 0),  # w-
    (0, 0, 0, 0, 1, 0),   # v+
    (0, 0, 0, 0, -1, 0),  # v-
    (0, 0, 0, 0, 0, 1),   # u+
    (0, 0, 0, 0, 0, -1),  # u-
]

# Relative directions mapping
REL_DIRECTIONS = {
    dir: [
        i for i in range(12) if i != BASIS_DIRECTIONS.index(tuple(-d for d in dir))
    ] for dir in BASIS_DIRECTIONS
}
