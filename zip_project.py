import os
import zipfile
import time
from tqdm import tqdm

def zip_project():
    items_to_zip = ["config.py", "main.py", "requirements.txt", "data", "data_utils", "inference", "models", "training", "xai"]
    output_filename = "deepfake__1.zip"
    
    print("Counting files to zip... This might take a moment.")
    total_files = 0
    valid_items = []
    
    for item in items_to_zip:
        if not os.path.exists(item):
            print(f"Warning: {item} not found, skipping.")
            continue
            
        valid_items.append(item)
        if os.path.isfile(item):
            total_files += 1
        elif os.path.isdir(item):
            for root, dirs, files in os.walk(item):
                total_files += len(files)
                
    print(f"Found {total_files} files. Creating {output_filename}...")
    
    start_time = time.time()
    
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        with tqdm(total=total_files, desc="Zipping", unit="file") as pbar:
            for item in valid_items:
                if os.path.isfile(item):
                    zipf.write(item)
                    pbar.update(1)
                elif os.path.isdir(item):
                    for root, dirs, files in os.walk(item):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, start=os.curdir)
                            zipf.write(file_path, arcname)
                            pbar.update(1)
                
    elapsed = time.time() - start_time
    print(f"\nSuccess! Created {output_filename} in {elapsed:.1f} seconds.")

if __name__ == "__main__":
    zip_project()
