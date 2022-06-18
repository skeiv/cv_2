import cv2
import pandas as pd
import os

from vehicle_detector import VehicleDetector

vd = VehicleDetector()
n = int(open("image_counter.txt", "r").read())

def reading(file_path, h):
    data = pd.read_csv(file_path, sep=",", header = h)
    return data

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def merge_channels(input_dir, output_dir):
    images = load_images_from_folder(input_dir)
    for i in range(0, n):
        b, a, a = cv2.split(images[i*3])
        a, r, a = cv2.split(images[i*3+1])
        a, a, g = cv2.split(images[i*3+2])
        img = cv2.merge((b, r, g))
        cv2.imwrite(os.path.join(output_dir, str(i+1) + ".jpg"), img)

def find_car(input_dir, output_cars):
    images = load_images_from_folder(input_dir)
    if len(vd.detect_vehicles(images[0])) == 0:
        rows = pd.DataFrame(["00001.jpg,False"])
    else:
        rows = pd.DataFrame(["00001.jpg,True"])
    for i in range(1, n):
        vehicle_boxes = vd.detect_vehicles(images[i])
        vehicle_count = len(vehicle_boxes)
        if (i + 1 < 10):
            name = '0000' + str(i+1) + '.jpg'
        elif (i + 1 < 100):
            name = '000' + str(i+1) + '.jpg'
        else:
            name = '00' + str(i+1) + '.jpg'
        if (vehicle_count == 0):
            flag = 'False'
        else:
            flag = 'True'
        row = pd.DataFrame([name + ',' + flag])
        rows = pd.concat([rows, row])
    rows.to_csv(output_cars, header=False, index=False)


#merge_channels(r"data", r"output")
find_car(r"output", "output.csv")

