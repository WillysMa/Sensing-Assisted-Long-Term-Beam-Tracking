#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Mengyuan Ma
@contact: mamengyuan410@gmail.com
@file: CSV_process.py
@time: 2025/12/16 21:01
"""

import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm
from Radar_KPI import *

# FFT parameters
FFT_TUPLE = (64, 256, 128)  # (fft_angle, fft_range, fft_velocity)


def process_radar_and_create_new_csv(csv_path, data_root, output_csv_path=None, output_suffix='FFT',
                                      test_mode=False, test_portion=0.01):
    """
    Process all radar data in the CSV file with FFT and create new files.
    
    Args:
        csv_path: Path to the original CSV file
        data_root: Root directory of the dataset
        output_csv_path: Path for the new CSV file (default: uses output_suffix)
        output_suffix: Suffix to add to the new CSV filename (default: 'FFT')
        test_mode: If True, only process a portion of the data (default: False)
        test_portion: Portion of data to process in test mode (default: 0.1 = 10%)
    
    Returns:
        DataFrame with updated radar paths
    """
    # Read the original CSV
    df = pd.read_csv(csv_path)
    total_rows = len(df)
    print(f"Loaded CSV with {total_rows} rows")
    
    # Apply test mode - only process a portion of the data
    if test_mode:
        num_rows = max(1, int(total_rows * test_portion))
        df = df.head(num_rows)
        print(f"TEST MODE: Processing only {num_rows} rows ({test_portion*100:.1f}% of data)")
    
    # Find all radar columns
    radar_columns = [col for col in df.columns if 'radar' in col.lower() and 'unit' in col.lower()]
    print(f"Found radar columns: {radar_columns}")
    
    # Create output directory for FFT processed radar data
    fft_output_dir = os.path.join(data_root, 'unit1', f'radar_data_{output_suffix}')
    os.makedirs(fft_output_dir, exist_ok=True)
    print(f"Output directory: {fft_output_dir}")
    
    # Keep track of processed files to avoid duplicates
    processed_files = {}
    
    # Create a copy of the dataframe for the new CSV
    df_new = df.copy()
    
    # Process each radar column
    for radar_col in radar_columns:
        print(f"\nProcessing column: {radar_col}")
        
        for idx in tqdm(range(len(df)), desc=f"Processing {radar_col}"):
            radar_path = df.loc[idx, radar_col]
            
            # Skip if path is NaN or invalid
            if pd.isna(radar_path) or radar_path == -99:
                continue
            
            # Get the original file name
            original_filename = os.path.basename(radar_path)
            
            # Create new filename with _FFT suffix
            name_without_ext = os.path.splitext(original_filename)[0]
            new_filename = f"{name_without_ext}_{output_suffix}.npy"
            new_filepath = os.path.join(fft_output_dir, new_filename)
            
            # Check if already processed
            if original_filename not in processed_files:
                # Construct full path to original radar file
                # Handle path format (may start with '/' or './')
                if radar_path.startswith('/') or radar_path.startswith('./'):
                    full_radar_path = os.path.join(data_root, radar_path.lstrip('./').lstrip('/'))
                else:
                    full_radar_path = os.path.join(data_root, radar_path)
                
                # Load and process radar data
                try:
                    smp_radar = loadmat(full_radar_path)['data']
                    radar_cube = Radar_Cube(smp_radar, FFT_TUPLE, remove_mean=True)
                    range_angle_map = Range_Angle(radar_cube, mean=True, log_scale=True)
                    range_doppler_map = Range_Doppler(radar_cube, mean=True, log_scale=True)
                    doppler_angle_map = Doppler_Angle(radar_cube, mean=True, log_scale=True)
                    
                    # Save the processed radar cube as numpy array
                    if output_suffix == 'RA':
                        np.save(new_filepath, range_angle_map)
                    elif output_suffix == 'RD':
                        np.save(new_filepath, range_doppler_map)
                    elif output_suffix == 'DA':
                        np.save(new_filepath, doppler_angle_map)
                    else:
                        np.save(new_filepath, radar_cube)
                    processed_files[original_filename] = new_filepath
                    
                except Exception as e:
                    print(f"\nError processing {full_radar_path}: {e}")
                    continue
            
            # Update the path in the new dataframe
            # Create relative path similar to original format
            new_relative_path = f"/unit1/radar_data_{output_suffix}/{new_filename}"
            df_new.loc[idx, radar_col] = new_relative_path
    
    # Save the new CSV
    if output_csv_path is None:
        csv_dir = os.path.dirname(csv_path)
        csv_name = os.path.basename(csv_path)
        name_without_ext = os.path.splitext(csv_name)[0]
        output_csv_path = os.path.join(csv_dir, f"{name_without_ext}_{output_suffix}.csv")
    
    df_new.to_csv(output_csv_path, index=False)
    print(f"\nNew CSV saved to: {output_csv_path}")
    print(f"Total unique radar files processed: {len(processed_files)}")
    
    return df_new


def main():
    # Configuration - modify these paths as needed
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    
    # Dataset paths
    data_root = os.path.join(current_dir, 'dataset', 'scenario9')
    csv_path = os.path.join(data_root, 'scenario9.csv')
    
    # Output CSV name suffix (change this to customize the output filename)
    # e.g., 'FFT', 'RA', 'RD', 'DA'
    OUTPUT_SUFFIX = 'RA'
    
    
    # Test mode configuration
    TEST_MODE = False  # Set to True to process only a portion of data
    TEST_PORTION = 0.1  # 0.1 = 10%, 0.01 = 1%
    
    # Check if files exist
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        print("Please update the csv_path variable with the correct path.")
        return
    
    print(f"Data root: {data_root}")
    print(f"CSV path: {csv_path}")
    print(f"FFT parameters: {FFT_TUPLE}")
    print(f"Output CSV suffix: {OUTPUT_SUFFIX}")
    print(f"Test mode: {TEST_MODE} (portion: {TEST_PORTION*100:.1f}%)" if TEST_MODE else "Test mode: OFF (full processing)")
    print("-" * 50)
    
    # Process radar data and create new CSV
    df_new = process_radar_and_create_new_csv(
        csv_path, data_root, 
        output_suffix=OUTPUT_SUFFIX,
        test_mode=TEST_MODE,
        test_portion=TEST_PORTION
    )
    
    # Display sample of the new dataframe
    print("\nSample of new CSV (first 5 rows, radar columns):")
    radar_cols = [col for col in df_new.columns if 'radar' in col.lower()]
    print(df_new[radar_cols].head())


    OUTPUT_SUFFIX = 'DA'
    # Process radar data and create new CSV
    df_new = process_radar_and_create_new_csv(
        csv_path, data_root, 
        output_suffix=OUTPUT_SUFFIX,
        test_mode=TEST_MODE,
        test_portion=TEST_PORTION
    )
    
    # Display sample of the new dataframe
    print("\nSample of new CSV (first 5 rows, radar columns):")
    radar_cols = [col for col in df_new.columns if 'radar' in col.lower()]
    print(df_new[radar_cols].head())


if __name__ == '__main__':
    main()
