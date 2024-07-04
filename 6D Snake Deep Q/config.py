import pygame
import math
import time
import librosa
import numpy as np
from math import *

song_path = "music/audio.ogg"
play_song = True

# Load audio and extract features
y, sr = librosa.load(song_path)

class ParamSpace:
    def __init__(self):
        self.sim_start = time.time()
        
        # Extract audio features
        print('loading amplitude')
        self.amplitude = np.abs(y)
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
        
        # New audio features
        print('loading pitch')
        self.pitch = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))[0]
        self.pitch_avg = np.nanmean(self.pitch)
        #self.harmonics = librosa.harmonic(y)
        print('loading formants')
        self.formants = librosa.lpc(y, order=8)
        print('loading mfccs')
        self.mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Create time arrays
        print('loading t_amplitude')
        self.t_amplitude = np.arange(len(self.amplitude)) / sr
        print('loading t_features')
        self.t_features = librosa.frames_to_time(np.arange(len(self.rms)), sr=sr)

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
        return self.interpolate(t, self.t_amplitude, self.amplitude)

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

    # New methods for additional audio features
    def get_pitch(self, t):
        pitch = self.interpolate(t, self.t_features, self.pitch)
        if np.isnan(pitch):
            return self.pitch_avg
        return pitch

    #def get_harmonics(self, t):
    #    idx = np.searchsorted(self.t_amplitude, t, side="right") - 1
    #    return self.harmonics[idx]

    def get_formants(self, t):
        idx = np.searchsorted(self.t_amplitude, t, side="right") - 1
        return self.formants[idx]

    def get_mfccs(self, t):
        idx = np.searchsorted(self.t_features, t, side="right") - 1
        return self.mfccs[:, idx]

# Define new visual parameters based on the new audio features
def get_pitch_color(param_space):
    t = param_space.get_time()
    pitch = param_space.get_pitch(t)
    normalized_pitch = (pitch - librosa.note_to_hz('C2')) / (librosa.note_to_hz('C7') - librosa.note_to_hz('C2'))
    r = int(255 * normalized_pitch)
    g = int(255 * (1 - normalized_pitch))
    b = int(128 + 127 * math.sin(normalized_pitch * math.pi))
    return (r, g, b)

#def get_harmonic_intensity(param_space):
#    t = param_space.get_time()
#    harmonics = param_space.get_harmonics(t)
#    return np.mean(harmonics)

def get_formant_effect(param_space):
    t = param_space.get_time()
    formants = param_space.get_formants(t)
    return np.sum(formants[:3])  # Use the first three formants

def get_mfcc_modulation(param_space):
    t = param_space.get_time()
    mfccs = param_space.get_mfccs(t)
    return np.mean(mfccs)




MODEL_SAVE_INTERVAL = 100000 # steps

# DQN Hyperparameters
BATCH_SIZE = 256
GAMMA = 0.95
EPSILON = 1 #0.5
EPSILON_MIN = 0.000001
EPSILON_STEPS = 100_000_000
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

MS_PER_FRAME = 600

# Screen dimensions
WIDTH, HEIGHT = 900, 900
GRID_SIZE = 10
NUM_GRIDS_W = 4
NUM_GRIDS_V = 4
NUM_GRIDS_U = 4

APPLE_COUNT = 1

# Define ZOOM and ROT1 as functions
def get_zoom(param_space):
    t = param_space.get_time()
    pitch = param_space.get_pitch(t)/param_space.pitch_avg
    print(t, pitch)
    return 100*(pitch**3)

def get_rot1(param_space):
    t = param_space.get_time()
    return t*0.1 + 0.05*sin(t)

# Define ROT2, ROT_CUBE1, ROT_CUBE2 as functions
def get_rot2(param_space):
    t = param_space.get_time()
    return 0.1 * t + 0.03 * math.cos(t * 0.03)

def get_rot_cube1(param_space):
    t = param_space.get_time()
    return 0.1 * t + 0.02*math.sin(t * 0.02)

def get_rot_cube2(param_space):
    t = param_space.get_time()
    return 0.1 * t + 0.06*math.cos(t * 0.04)

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Define color configurations as functions
def get_grid_color(param_space):
    t = param_space.get_time()
    chroma = param_space.get_chroma(t)
    r = int(128 + 127 * chroma[0])
    g = int(128 + 127 * chroma[4])
    b = int(128 + 127 * chroma[7])
    return (r, g, b, 60)

SNAKE_FILL_COLOR = (0, 220, 0)

def get_snake_line_color(param_space):
    t = param_space.get_time()
    g = int(200 + 55 * param_space.get_spectral_centroid(t) / 1000)
    b = int(100 + 155 * param_space.get_rms(t))
    return (0, g, b)

APPLE_FILL_COLOR = (255, 0, 0, 255)

def get_apple_line_color(param_space):
    t = param_space.get_time()
    r = int(180 + 75 * param_space.get_spectral_bandwidth(t) / 1000)
    g = int(100 * param_space.get_zero_crossings(t))
    return (r, g, 0)

# Rendering configurations
SNAKE_CONNECTION_ALPHA = 0.9

# Define cell dimensions as functions
def get_cell_width(param_space):
    t = param_space.get_time()
    return 1.0 + 0.3 * param_space.get_rms(t) + 0.1 * math.sin(t * 0.1)

def get_cell_height(param_space):
    t = param_space.get_time()
    return 1.0 + 0.3 * param_space.get_zero_crossings(t) + 0.1 * math.cos(t * 0.1)

def get_cell_length(param_space):
    t = param_space.get_time()
    return 1.0 + 0.3 * param_space.get_spectral_centroid(t) / 1000 + 0.1 * math.sin(t * 0.15)

# New visual parameters
def get_main_grid_axes_color(param_space):
    t = param_space.get_time()
    chroma = param_space.get_chroma(t)
    r = int(200 + 55 * chroma[2])
    g = int(200 + 55 * chroma[6])
    b = int(200 + 55 * chroma[10])
    return (r, g, b, 40)

def get_subgrid_axes_color(param_space):
    t = param_space.get_time()
    chroma = param_space.get_chroma(t)
    r = int(180 + 75 * chroma[1])
    g = int(180 + 75 * chroma[5])
    b = int(180 + 75 * chroma[9])
    return (r, g, b, 180)

def get_main_grid_line_thickness(param_space):
    t = param_space.get_time()
    return 2.0 + 1.5 * param_space.get_rms(t) + 0.5 * math.sin(t * 0.2)

def get_subgrid_line_thickness(param_space):
    t = param_space.get_time()
    return 1.0 + 0.8 * param_space.get_zero_crossings(t) + 0.3 * math.cos(t * 0.2)

def get_snake_line_thickness(param_space):
    t = param_space.get_time()
    return 2.0 + 1.5 * param_space.get_spectral_centroid(t) / 1000 + 0.5 * math.sin(t * 0.25)

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
