from pathlib import Path
import os
# this code eliminates too small images
cropped_dir = "cropped\\"
cropped_files = os.listdir(cropped_dir)

size_lst = []
for file in cropped_files:
    file_size =Path(cropped_dir + file).stat().st_size
    if file_size < 10000:
        os.remove(cropped_dir + file)
