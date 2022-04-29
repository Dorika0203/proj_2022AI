import shutil
import os

# this code makes folder and sort pictures by car name
def find_nth(string, substring, n):
   if (n == 1):
       return string.find(substring)
   else:
       return string.find(substring, find_nth(string, substring, n - 1) + 1)

origin_file_path = "cropped\\"
dir_list = os.listdir(origin_file_path)
output_path = "output\\"

# make folder and copy pictures
previous_tag = ""
cnt = 0
all_cnt = 0
for dir in dir_list:
    divide_loc = find_nth(dir, '$', 4)
    car_tag = (dir[0:divide_loc])
    car_tag = car_tag.replace("$", "\\")
    if car_tag != previous_tag:
        previous_tag = car_tag
        output_dir = output_path + car_tag
        os.makedirs(output_dir, exist_ok=True)
        all_cnt = all_cnt + cnt
        print(all_cnt)
        cnt = 0
    shutil.copyfile(origin_file_path + dir, output_path + car_tag + "\\" + str(cnt) + ".jpg")
    cnt += 1
