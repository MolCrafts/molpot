#!/usr/bin/env python3
import sys
sys.path.insert(0, '../../../')  # Go to src/ directory

from molpot.pipeline.qdpi import QDpi

print("=== CHECKING QDpi DATASET CONFIGURATION ===")

qdpi = QDpi(subset='all', save_dir='./data/qdpi')
ds = qdpi.get_subset_data()

print(f"\nTOTAL CATEGORIES: {len(ds)}")
total_files = 0

for category, subds in ds.items():
    print(f"\n{category.upper()} CATEGORY ({len(subds)} files):")
    for key, url in subds.items():
        full_url = f"https://gitlab.com/RutgersLBSR/QDpiDataset/-/raw/main/data/{url}"
        local_file = f"./data/qdpi/{key}.hdf5"
        
        # Check if file exists
        import os
        exists = "✅ EXISTS" if os.path.exists(local_file) else "❌ MISSING"
        
        print(f"  {key:15} -> {url:25} {exists}")
        total_files += 1

print(f"\nTOTAL FILES TO PROCESS: {total_files}")
print(f"FULL GITLAB ROOT: https://gitlab.com/RutgersLBSR/QDpiDataset/-/raw/main/data/")

# Check which files need downloading
import os
missing_files = []
for category, subds in ds.items():
    for key, url in subds.items():
        local_file = f"./data/qdpi/{key}.hdf5"
        if not os.path.exists(local_file):
            missing_files.append((key, url))

print(f"\nFILES TO DOWNLOAD ({len(missing_files)}):")
for key, url in missing_files:
    print(f"  {key}: {url}") 