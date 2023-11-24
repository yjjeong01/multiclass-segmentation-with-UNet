import numpy as np
import cv2
import os

# Checking num of classes for all sequences in Grayscale

folder_path = r'../dataset/train'
subfolder_list = []
for dir_name in os.listdir(folder_path):
    subfolder_list.append(dir_name)

class_list = []
for folder_name in subfolder_list:
    folder = os.path.join(folder_path, folder_name)
    img_folder_path = os.path.join(folder, "mask")
    img_files = os.listdir(img_folder_path)

    num_class_list = []

    for file in img_files:
        img_path = os.path.join(img_folder_path, file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_np = np.array(img)

        colors = np.unique(img_np)
        colors = np.delete(colors, 0)

        for i in colors:
            if i not in class_list:
                class_list.append(i)

print(class_list)
