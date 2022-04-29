from PIL import Image
import os
# this code crops the picture by label

label_dir = "labels\\"
label_files = os.listdir(label_dir)

for label_file in label_files:
    f = open(label_dir + label_file)
    biggest_val = 0.0
    car_leftTop = [0.0,0.0]
    car_rightBot = [0.0,0.0]
    #format: class_id center_x center_y width height
    while True:
        line = f.readline()
        if not line: break
        line = line.split()
        tag = []
        for item in line:
            tag.append(float(item))

        if tag[3] * tag[4] > biggest_val:
            biggest_val = tag[3] * tag[4]
            car_leftTop[0] = tag[1] - tag[3]/2
            car_rightBot[0] = tag[1] + tag[3]/2
            car_leftTop[1] = tag[2] - tag[4]/2
            car_rightBot[1] = tag[2] + tag[4]/2
    f.close()

    picture_dir = label_file.replace("txt", "JPG")
    img = Image.open(picture_dir)
    car_leftTop[0] = img.size[0] * car_leftTop[0]
    car_rightBot[0] = img.size[0] * car_rightBot[0]
    car_leftTop[1] = img.size[1] * car_leftTop[1]
    car_rightBot[1] = img.size[1] * car_rightBot[1]
    xy = (car_leftTop[0], car_leftTop[1], car_rightBot[0], car_rightBot[1])
    crop_img = img.crop(xy)
    #make the 'cropped' folder previously
    crop_img.save("cropped\\" + picture_dir)