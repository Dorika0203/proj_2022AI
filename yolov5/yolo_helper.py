import os
from os import path
import shutil
# this code makes label data using pre-trained yolo5 model
dir_list = []
jpg_list = []

# get all the path info of directory and jpg files in "kcar"
def search_fileNpath_in_dir(root_dir, prefix):
    files = os.listdir(root_dir)
    for file in files:
        path = os.path.join(root_dir, file)
        if path.endswith('.jpg') or path.endswith('.JPG') : jpg_list.append(prefix + path)
        else: dir_list.append(prefix + path)

        if os.path.isdir(path):
            search_fileNpath_in_dir(path, prefix + "")

search_fileNpath_in_dir("kcar", "")

# copy all img files to data\images\
for jpg in jpg_list:
    file_name = jpg.replace("\\","$")
    origin_path = jpg
    copy_dir = "data\\images\\"
    if not path.exists(copy_dir + file_name):
        shutil.copy(origin_path, copy_dir + file_name)

os.system("python detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source data/images --save-txt")


# and take the labels file to data\\images folder
# we'll process photo cutting in the directory