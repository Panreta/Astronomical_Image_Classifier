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


phot_root = r"/home/share/guofangkeda/wangcunshi/LSST/ilpart/50183704.csv"
# record_root = r"/mnt/nodestor/home/sincloud_user/Cunshi/Tingyu/LSST/DATA/Record/Lomb-Scargle Periodograms"
record_root = r"/home/share/guofangkeda/wangcunshi/LSST/ilpart"



def process_file(file_path,output_subfolder):
    try:
        df = pd.read_csv(file_path)

        # Clean data
        df = df.dropna()
        df["MJD"] = pd.to_numeric(df["MJD"])
        df["FLUXCAL"] = pd.to_numeric(df["FLUXCAL"])
        df["FLUXCALERR"] = pd.to_numeric(df["FLUXCALERR"])
        df = df.dropna(subset=["MJD", "FLUXCAL", "FLUXCALERR"], how='any')
        df["BAND"] = df["BAND"].astype(str).str.strip()
        existing_bands = df["BAND"].unique()

        # generate folder
        file_id = os.path.splitext(os.path.basename(file_path))[0]
        file_output_folder = os.path.join(output_subfolder, file_id)# AGN 001 8094
        os.makedirs(file_output_folder, exist_ok=True)

        for band in existing_bands:
            band_data = df[
                        (df["BAND"] == band) &
                        (df["detect"] == 1) &
                        (df["FLUXCALERR"] > 0)&
                        (df["MJD"] > 0) &
                        (df["MJD"] <= 10)
                    ] 
            
            band_data.sort_values("MJD", inplace=True)
            print(band_data)


            # .value: returns the underlying NumPy array

            time = band_data["MJD"].to_numpy(dtype=float) # .value: returns the underlying NumPy array
            flux = band_data["FLUXCAL"].to_numpy()
            flux_err = band_data["FLUXCALERR"].to_numpy()
            time_seconds = (time - np.min(time)) * 86400.0  # day->second,start from 0
            print(time_seconds)
            
            # Add small jitter to prevent zero time differences
            time_seconds += np.random.normal(0, 0.1, len(time_seconds))  # 100ms jitter
                
            # Calculate time differences and frequency
            time_diffs = np.diff(time_seconds)
            print(time_diffs)
            valid_diffs = time_diffs[time_diffs > 0]
          
            
        

            
            if len(time_seconds) < 1:
                print(f"Skipping {file_id} band={band}: no positive time differences")
                continue

         


            median_diff = np.median(valid_diffs)
            total_time = np.max(time_seconds) - np.min(time_seconds)
            min_freq = 1/(10*total_time)  # Detect periods up to 10x observation baseline
            max_freq = 1/(0.5*median_diff)  # Where median_diff = np.median(time_diffs)
            max_freq = min(max_freq, 1e4)  # Cap at 10kHz (adjust per your needs)
            min_freq = max(min_freq, 1e-7)  # Minimum 0.1μHz

            print("time_seconds:", time_seconds[:5], "...", time_seconds[-5:])


            if not (np.isfinite(min_freq) and np.isfinite(max_freq)):
                print(f"Bad freq bounds: min={min_freq}, max={max_freq}")
                continue


            # Calculate periodogram
            ls = LombScargle(time_seconds, flux, flux_err)
            print(vars(ls))  
            # all good
            
            frequency, power = ls.autopower(minimum_frequency=min_freq,maximum_frequency=max_freq)# problem
            print(1)

            
            best_period_seconds = 1 / frequency[np.argmax(power)]
            print(2)
            best_period_days = best_period_seconds / 86400  # Convert back to days
            print(3)

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
        print(f"\n❌ ERROR processing {file_path}: {e}")


if __name__ == '__main__':
    process_file(phot_root, record_root)



