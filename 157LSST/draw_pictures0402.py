from multiprocessing import Pool, cpu_count

import os

import pandas as pd

import matplotlib.pyplot as plt

from astropy.timeseries import LombScargle

import numpy as np

import shutil

  

# Mapping of bands to colors

band_colors = {

"u": "#1f77b4",

"g": "#2ca02c",

"r": "#d62728",

"i": "#9467bd",

"z": "#8c564b",

"Y": "#e377c2"

}

  
  

phot_root = r"/mnt/nodestor/home/sincloud_user/Cunshi/Tingyu/LSST/DATA/Type/PhotSplit"

# record_root = r"/mnt/nodestor/home/sincloud_user/Cunshi/Tingyu/LSST/DATA/Record/Lomb-Scargle Periodograms"

record_root = r"/mnt/nodestor/home/sincloud_user/Cunshi/Tingyu/LSST/DATA/Record/LS_Test"

  

# if exist, clean the original one

if os.path.exists(record_root):

    shutil.rmtree(record_root)

os.makedirs(record_root, exist_ok=True)

  

def process_file(file_path,output_subfolder):

    try:

        df = pd.read_csv(file_path)

        record_root = output_subfolder

        df = df.dropna()

        df["MJD"] = pd.to_numeric(df["MJD"])

        df["FLUXCAL"] = pd.to_numeric(df["FLUXCAL"])

        df["FLUXCALERR"] = pd.to_numeric(df["FLUXCALERR"])

        df = df.dropna(subset=["MJD", "FLUXCAL", "FLUXCALERR"], how='any')

        df["BAND"] = df["BAND"].astype(str).str.strip()

        existing_bands = df["BAND"].unique()

  
# generate folder

        file_id = os.path.splitext(os.path.basename(file_path))[0]

        file_output_folder = os.path.join(record_root, file_id)# AGN 001 8094

        os.makedirs(file_output_folder, exist_ok=True)

  

        for band in existing_bands:

            band_data = df[(df["BAND"] == band) & (df["detect"] == 1)& (df["FLUXCALERR"] > 0)]

  

            # .value: returns the underlying NumPy array

            time = df["MJD"].values# .value: returns the underlying NumPy array

            flux = df["FLUXCAL"].values

            flux_err = df["FLUXCALERR"].values

            time_seconds = (time - np.min(time)) * 86400 # day->second,start from 0
            # Add small jitter to prevent zero time differences

            time_seconds += np.random.normal(0, 0.1, len(time_seconds))  # 100ms jitter

            # Calculate time differences and frequency
            time_diffs = np.diff(np.sort(time_seconds))

            valid_diffs = time_diffs[time_diffs > 0]

  
  

            median_diff = np.median(valid_diffs)

            total_time = np.max(time_seconds) - np.min(time_seconds)

            min_freq = 1/(10*total_time)  # Detect periods up to 10x observation baseline

            max_freq = 1/(0.5*median_diff)  # Where median_diff = np.median(time_diffs)

            max_freq = min(max_freq, 1e4)  # Cap at 10kHz (adjust per your needs)
            min_freq = max(min_freq, 1e-7)  # Minimum 0.1μHz

            # print(time_diffs)

            # print(min_diff)# 0.00018891405738941103

            # print(max_freq)# 52934.1232631877

            # print(min_freq)# 0.00011574020880492687

  
  

            # Calculate periodogram

            ls = LombScargle(time_seconds, flux, flux_err)

            frequency, power = ls.autopower(minimum_frequency=min_freq,maximum_frequency=max_freq)

  

            best_period_seconds = 1 / frequency[np.argmax(power)]

            best_period_days = best_period_seconds / 86400  # Convert back to days

  

            # Generate plot

            plt.figure(figsize=(10, 5))

            plt.plot(1/frequency/86400, power, color=band_colors[band])

            plt.xlabel("Period (days)")

            plt.ylabel("Power")

            plt.title(f"{file_id} - Band {band}\nBest Period: {best_period_days:.4f} days")

            plt.grid(True)

            plot_path = os.path.join(file_output_folder, f"{band}_periodogram.png")

            plt.savefig(plot_path, dpi=150, bbox_inches='tight')

            plt.close()

  

            print("\n=== RESULTS ===")

            print(f"Best period: {best_period_days:.6f} days")

            print(f"Plot saved to: {plot_path}")

  

    except Exception as e:

        print(f"\n❌ ERROR: {str(e)}")

        input("Error occur!!!!")

  
  

if __name__ == '__main__':

    for folder_name in os.listdir(phot_root):

        folder_path = os.path.join(phot_root, folder_name)#AGN

        output_folder = os.path.join(record_root, folder_name)

        os.makedirs(output_folder, exist_ok=True)

  

        for subfolder_name in os.listdir(folder_path):

            subfolder_path = os.path.join(folder_path, subfolder_name)#AGN 001

            output_subfolder = os.path.join(output_folder, subfolder_name)

            os.makedirs(output_subfolder, exist_ok=True)

  

            for file_name in os.listdir(subfolder_path):

                file_path = os.path.join(subfolder_path, file_name)#AGN 001 8094.CSV

                process_file(file_path, output_subfolder)

  

    # # Sequential processing

    # for file_path, output_subfolder in tasks:

    #     process_file(file_path, output_subfolder)