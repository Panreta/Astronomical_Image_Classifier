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

# Root paths
phot_root =r"/home/wcs/LSST/PhotSplit"
record_root = r"/home/wcs/LSST/LombScargle"

# Make sure the output root exists
if os.path.exists(record_root):
    shutil.rmtree(record_root)
os.makedirs(record_root, exist_ok=True)

def process_file(file_path, output_subfolder):
    try:
        df = pd.read_csv(file_path).dropna()
    
        
        # Clean and validate data types (fix for NaN/invalid values)
        df["MJD"] = pd.to_numeric(df["MJD"])
        df["FLUXCAL"] = pd.to_numeric(df["FLUXCAL"])
        df["FLUXCALERR"] = pd.to_numeric(df["FLUXCALERR"])
        df = df.dropna(subset=["MJD", "FLUXCAL", "FLUXCALERR"], how='any')  # Drop rows with NaN

        df["BAND"] = df["BAND"].astype(str).str.strip()
        filtered_df = df.copy()  # For simplicity, we skip the 3-sigma cleaning here
        if filtered_df.empty:
            print(f"⚠️ No data left after filtering in {file_path}")
            return

        existing_bands = filtered_df["BAND"].unique()
        file_id = os.path.splitext(os.path.basename(file_path))[0]
        file_output_folder = os.path.join(output_subfolder, file_id)
        os.makedirs(file_output_folder, exist_ok=True)

        for band in existing_bands:
            band_data = filtered_df[(filtered_df["BAND"] == band) & (filtered_df["detect"] == 1) &  (filtered_df["FLUXCALERR"] > 0)]
            if band_data.empty:
                continue
            
            time = band_data["MJD"].values
            flux = band_data["FLUXCAL"].values
            flux_err = band_data["FLUXCALERR"].values
            # 筛选出 MJD < 10 的数据行
            mask = time < 10.0001
            time = time[mask]
            flux = flux[mask]
            flux_err = flux_err[mask]
            
            time_seconds = (time - np.min(time)) * 86400
            time_seconds += np.random.normal(0, 30, len(time_seconds))
            time_diffs = np.diff(np.sort(time_seconds))
            valid_diffs = time_diffs[time_diffs > 0]
            median_diff = np.median(valid_diffs)
            total_time = np.max(time_seconds) - np.min(time_seconds)
            min_freq = max(1/(10*total_time), 1e-7)
            max_freq = min(1/(0.5*median_diff), 1e4)

            try:
                ls = LombScargle(time_seconds, flux, flux_err)
                frequency, power = ls.autopower(minimum_frequency=min_freq,maximum_frequency=max_freq)
                best_period_seconds = 1 / frequency[np.argmax(power)]
                best_period = best_period_seconds / 86400

                plt.figure(figsize=(10, 5))
                plt.plot(1 / frequency /86400, power, color=band_colors.get(band, "gray"))
                plt.xlabel("Period (days)")
                plt.ylabel("Power")
                plt.title(f"{file_id} - Band {band} (Best Period: {best_period:.2f}d)")
                plt.grid(True)
                plt.tight_layout()

                plot_filename = f"{band}_periodogram.png"
                plot_path = os.path.join(file_output_folder, plot_filename)
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()

            except Exception as e:
                print(f"❌ Failed periodogram for {file_id} [{band}]: {e}")

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")


if __name__ == '__main__':
    # ... [之前的遍历文件夹和创建目录的逻辑保持不变]

    # 使用所有可用CPU核心数的一半作为进程池大小
    num_processes = cpu_count() // 2
    pool = Pool(processes=num_processes)

    tasks = []
    for folder_name in os.listdir(phot_root):
        folder_path = os.path.join(phot_root, folder_name)
        output_folder = os.path.join(record_root, folder_name)
        os.makedirs(output_folder, exist_ok=True)

        for subfolder_name in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder_name)
            output_subfolder = os.path.join(output_folder, subfolder_name)
            os.makedirs(output_subfolder, exist_ok=True)

            for file_name in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file_name)
                tasks.append((file_path, output_subfolder))

    # 并行处理所有任务
    pool.starmap(process_file, tasks)

    pool.close()
    pool.join()