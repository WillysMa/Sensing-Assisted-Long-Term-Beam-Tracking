import numpy as np
import torch

class Radar_KPI: ## FMCW radar
    def __init__ (self):
        # Radar configuration
        self.N_r = 4 # number of receiver antennas

        # Frame configuration
        self.S = 15 # chirp slope (MHz/us)
        self.f_0 = 77 # starting frequency (GHz)
        self.F_s = 5 # sampling frequency (samples/us)
        self.N_sample = 256 # number of ADC samples per chirp
        self.N_chirp = 128 # number of chirp loops per frame
        self.T_PRI = 65 # chirp repetition interval (us)
        
        # Derived parameters
        self.T_active = self.N_sample / self.F_s # active chirp duration (us)
        self.BW = self.S * self.T_active # chirp bandwidth (MHz)
        self.f_c = self.f_0 + 1e-3 * self.BW / 2 # center frequency (GHz)
        
        # Range KPI
        self.range_res = 3 * 1e8 / (2 * self.BW * 1e6) # range resolution (m)
        self.range_max = self.N_sample * self.range_res # maximum unambiguous detectable range (m)
        
        # Doppler KPI
        self.velocity_res = (3 * 1e8 / (self.f_c * 1e9)) / (2 * self.T_PRI * 1e-6 * self.N_chirp) * 3.6 # velocity resolution (km/hr)
        self.velocity_max = (3 * 1e8 / (self.f_c * 1e9)) / (4 * self.T_PRI * 1e-6) * 3.6 # maximum unambiguous detectable velocity (km/hr)

        # Codebook configuration

        self.measurement_offset_angle = 4 * np.pi / 180
        self.angle_start = 0 - self.measurement_offset_angle
        self.angle_end = np.pi - self.measurement_offset_angle
        self.num_of_angle = 64
        # self.angle_of_beams = 90 - np.arange(self.angle_start, self.angle_end, (self.angle_end - self.angle_start) / self.num_of_angle)[np.argmax(self.codebook_pattern, axis=1)] / np.pi * 180 ## degree
    
    def print_KPI(self):
        print(f'Active chirp duration = {self.T_active} us\n',
              f'Chirp bandwidth = {self.BW} MHz\n',
              f'Center frequency = {self.f_c} GHz\n',
              f'Range resolution = {self.range_res} m\n',
              f'Maximum range = {self.range_max} m\n',
              f'Velocity resolution = {self.velocity_res} km/hr\n',
              f'Maximum velocity = {self.velocity_max} km/hr')

def Radar_Cube(radar_data, fft_tuple, remove_mean=True):
    """
    Perform range, angle, and velocity FFT on radar data.

    Args:
        data (np.ndarray): Input radar data of shape (N, S, A)
        fft_tuple (tuple): (fft_angle, fft_range, fft_velocity), the FFT sizes for each axis.

    Returns:
        torch.Tensor: The the processed radar data cube.
    """
    (fft_angle, fft_range, fft_velocity) = fft_tuple
    # Perform Range-DFT
    range_DFT = np.fft.fft(radar_data, n=fft_range, axis=1) ## [4, fft_range, 128]
    # Remove DC offset (relatively static objects)
    if remove_mean:
        range_DFT  = range_DFT - np.mean(range_DFT, axis=2, keepdims=True)
    # Perform Doppler-DFT
    doppler_DFT = np.fft.fft(range_DFT, n=fft_velocity, axis=2) ## [4, fft_range, fft_velocity]
    # Perform Angle-DFT (Radar Cube)
    angle_DFT = np.fft.fft(doppler_DFT, n=fft_angle, axis=0) ## [fft_angle, fft_range, fft_velocity]
    radar_cube = np.fft.fftshift(angle_DFT, axes=(0, 2))
    return radar_cube

def Range_Doppler(radar_cube, mean=True, log_scale=True):

    if mean:
        range_doppler_map = np.mean(np.abs(radar_cube), axis=0) ## [fft_range, fft_velocity]
    else:
        range_doppler_map = np.sum(np.abs(radar_cube), axis=0) ## [fft_range, fft_velocity]
    if log_scale:
        range_doppler_map = np.log2(1 + range_doppler_map)
    return range_doppler_map

def Range_Angle(radar_cube, mean=True, log_scale=True):
    if mean:
        range_angle_map = np.mean(np.abs(radar_cube), axis=2).T ## [fft_range, fft_angle]
    else:
        range_angle_map = np.sum(np.abs(radar_cube), axis=2).T ## [fft_range, fft_angle]
    if log_scale:
        range_angle_map = np.log2(1 + range_angle_map)
    return range_angle_map

def Doppler_Angle(radar_cube, mean=True, log_scale=True):
    if mean:
        doppler_angle_map = np.mean(np.abs(radar_cube), axis=1).T ## [fft_velocity, fft_range]
    else:
        doppler_angle_map = np.sum(np.abs(radar_cube), axis=1).T ## [fft_velocity, fft_range]
    if log_scale:
        doppler_angle_map = np.log2(1 + doppler_angle_map)
    return doppler_angle_map



