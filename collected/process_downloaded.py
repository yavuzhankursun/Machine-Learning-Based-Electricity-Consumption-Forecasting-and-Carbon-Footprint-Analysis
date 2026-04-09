"""
Read the climate_temperature_data.csv from Downloads folder
and copy to the target directory.
"""
import shutil
import os
import glob

src_patterns = [
    os.path.expanduser("~/Downloads/climate_temperature_data.csv"),
    os.path.expanduser("~/Downloads/climate_temperature_data (1).csv"),
    r"C:\Users\LENOVO\Downloads\climate_temperature_data.csv",
    r"C:\Users\LENOVO\Downloads\climate_temperature_data (1).csv",
]

dst = r"C:\Users\LENOVO\Desktop\Tez Yeni\data\collected\climate_temperature_data.csv"

for src in src_patterns:
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"Copied: {src} -> {dst}")
        # Verify
        with open(dst, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"Total lines: {len(lines)}")
            print(f"First 3 lines:")
            for line in lines[:3]:
                print(f"  {line.strip()}")
            print(f"Last 3 lines:")
            for line in lines[-3:]:
                print(f"  {line.strip()}")
        break
else:
    # List downloads directory
    dl = os.path.expanduser("~/Downloads")
    print(f"Downloads directory: {dl}")
    csv_files = glob.glob(os.path.join(dl, "*.csv"))
    print(f"CSV files found: {csv_files[:10]}")
    print("climate_temperature_data.csv not found in Downloads")
