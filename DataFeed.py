#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Mengyuan Ma
@contact: mamengyuan410@gmail.com
@file: DataFeed.py
@time: 2025/12/22 12:30
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transf
from skimage import io
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from Radar_KPI import *
from scipy.io import loadmat

def create_samples(root, portion=1.):
    f = pd.read_csv(root, na_values='')
    f = f.fillna(-99)
    Total_Num = len(f)
    num_data = int(Total_Num * portion)
    data_samples_rgb = []
    data_samples_radar = []
    pred_beam = []
    inp_beam = []
    for idx, row in f.head(num_data).iterrows():
        vision_data = row['camera1':'camera8'].tolist()
        data_samples_rgb.append(vision_data)
        radar_data = row['radar1':'radar8'].tolist()
        data_samples_radar.append(radar_data)

        # Dynamic approach: get all future_beam columns
        future_beam_cols = [col for col in f.columns if col.startswith('future_beam')]
        future_beam_cols.sort()  # Ensure consistent ordering (future_beam1, future_beam2, etc.)
        future_beam = row[future_beam_cols].tolist()
        pred_beam.append(future_beam)

        input_beam = row['beam1':'beam8'].tolist()

        inp_beam.append(input_beam)

    # print('list is ready')
    return data_samples_rgb, data_samples_radar, inp_beam, pred_beam


class DataFeed(Dataset):
    def __init__(self, data_root, root_csv, seq_len=8, transform=None,   
    fft_tuple=(64, 256,128), clipped_range=128, portion=1.):

        self.data_root = data_root
        self.samples_rgb, self.samples_radar, self.inp_val, self.pred_val = create_samples(root_csv, portion=portion)
        self.seq_len = seq_len
        self.transform = transform
        self.fft_tuple = fft_tuple
        self.clipped_range = clipped_range


    def __len__(self):
        return len(self.samples_rgb)

    def __getitem__(self, idx):
        samples_rgb = self.samples_rgb[idx]
        samples_radar = self.samples_radar[idx]
        beam_val = self.pred_val[idx]
        input_beam = self.inp_val[idx]


        # out_beam = torch.zeros((3,))
        image_val = np.zeros((self.seq_len, 224,224))
        image_dif = np.zeros((self.seq_len-1, 224, 224))
        image_motion_masks = np.zeros((self.seq_len - 1, 224, 224))

        beam_past = []
        clipped_range = self.clipped_range

        radar_val_range_angle = np.zeros((self.seq_len, clipped_range, self.fft_tuple[0]))
        radar_val_doppler_angle = np.zeros((self.seq_len, self.fft_tuple[2], self.fft_tuple[0]))
        radar_dif_RA = np.zeros((self.seq_len - 1, clipped_range, self.fft_tuple[0]))
        radar_dif_DA = np.zeros((self.seq_len - 1, self.fft_tuple[2], self.fft_tuple[0]))

        def _p(rel_path):
            # CSV paths start with '/', so join safely without duplicating separators
            return os.path.join(self.data_root, rel_path.lstrip("/"))

        for i, (smp_rgb_path,smp_radar_path) in enumerate(zip(samples_rgb,samples_radar)):
            beam_past_i = int(np.argmax(np.loadtxt(_p(input_beam[i])))) # start with 0
            beam_past.append(beam_past_i)
            # Load the image
            img = self.transform(io.imread(_p(smp_rgb_path)))
            # Load the radar
            range_angle_map = np.load(_p(smp_radar_path))

            range_angle_clipped = range_angle_map[:clipped_range, ...]
            smp_radar_path_DA = smp_radar_path.replace('_RA', '_DA')
            doppler_angle_map = np.load(_p(smp_radar_path_DA))
            # # Store the smoothed image
            radar_val_range_angle[i,...] = range_angle_clipped #/np.max(smp_radar[:clipped_range, ...]+ 1e-6) # normalize the radar data
            radar_val_doppler_angle[i,...] = doppler_angle_map #/np.max(smp_radar[:clipped_range, ...]+ 1e-6) # normalize the radar data


            img = rgb2gray(img)  # Convert to grayscale

            # Apply Gaussian filtering
            img_smoothed = gaussian_filter(img, sigma=1)  # Adjust sigma for smoothing strength

            # Store the smoothed image
            image_val[i, ...] = img_smoothed

            # Compute the difference with the previous frame
            if i >= 1:
                diff = np.abs(image_val[i, ...] - image_val[i - 1, ...])
                image_dif[i - 1, ...] = diff

                # Calculate the dynamic threshold as 10% of the maximum pixel value in the difference image
                max_pixel_value = np.max(diff)
                threshold_value = 0.1 * max_pixel_value

                # Generate binary mask of significant changes
                motion_mask = (diff > threshold_value).astype(np.uint8)
                image_motion_masks[i - 1, ...] = motion_mask
                #------------------------------------below is the radar part------------------------------------
                diff_radar_RA = np.abs(radar_val_range_angle[i,...] - radar_val_range_angle[i - 1,...])
                diff_radar_DA = np.abs(radar_val_doppler_angle[i,...] - radar_val_doppler_angle[i - 1,...])
                radar_dif_RA[i - 1, ...] = diff_radar_RA
                radar_dif_DA[i - 1, ...] = diff_radar_DA
        image_masks = torch.tensor(image_motion_masks,dtype=torch.float32)

        radar_RA = torch.tensor(radar_val_range_angle,dtype=torch.float32)
        radar_DA = torch.tensor(radar_val_doppler_angle,dtype=torch.float32)

        beam_future = []
        for i in range(len(beam_val)):
            beam_future_i = int(np.argmax(np.loadtxt(_p(beam_val[i])))) 
            beam_future.append(beam_future_i)

        input_beam = torch.tensor(beam_past,dtype=torch.int64)
        out_beam = torch.tensor(beam_future,dtype=torch.int64)
        pass
        return image_masks, radar_RA, radar_DA, input_beam.long(), torch.squeeze(out_beam.long())

