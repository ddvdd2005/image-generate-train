import json
import pathlib
import cv2
import numpy as np
import os
import random
from random import randint
from scipy import ndimage
from PIL import Image

image_files = []
background_files = []
for root, _, files in os.walk("./images"):
    for file in files:
        image_files.append(os.path.join(root, file))
for root, _, files in os.walk("./background"):
    for file in files:
        background_files.append(os.path.join(root, file))
image_files.sort()


def get_full_image(folder, image_nbr, coco_json):
    coco_json["images"].append({"file_name": f"{image_nbr:06d}.png", "height": 400, "width": 600, "id": image_nbr})
    bg_image = cv2.imread(random.choice(background_files), cv2.IMREAD_UNCHANGED)
    bg_image = cv2.resize(bg_image, (600, 400), interpolation=cv2.INTER_AREA)
    random_nbr = random.random()
    if random_nbr <= 0.1:
        nbr_images = 0
    elif random_nbr <= 0.75:
        nbr_images = 1
    else:
        nbr_images = 2
    bg_image = Image.fromarray(bg_image)
    saved = []
    for i in range(nbr_images):
        while True:
            anno_image, category_id = get_image()
            width = anno_image.shape[1]
            height = anno_image.shape[0]
            if nbr_images == 1:
                ratio = random.uniform(0.4, 0.8)
            else:
                ratio = random.uniform(0.4, 0.8)
            if width / 600 > height / 400:
                width, height = int(ratio * 600), int(height * ratio * 600 / width)
            else:
                height, width = int(ratio * 400), int(width * ratio * 400 / height)
            anno_image = cv2.resize(anno_image, (width, height), interpolation=cv2.INTER_AREA)
            anno_image = Image.fromarray(anno_image)
            min_x = int(0 - width / 3)
            max_x = int(600 - 2 * width / 3)
            min_y = int(0 - height / 3)
            max_y = int(400 - 2 * height / 3)
            x = int(random.uniform(min_x, max_x))
            y = int(random.uniform(min_y, max_y))
            if i == 0 and nbr_images == 1:
                break
            if i == 0 and nbr_images == 2:
                saved = [x, y, width, height]
                break
            if i == 1:
                overlap = calculate_overlap_percentage(x, y, width, height, saved[0], saved[1], saved[2], saved[3])
                if overlap <= 30:
                    break
        bg_image.paste(anno_image, (x, y), anno_image)
        global annotation_id
        coco_json["annotations"].append({"image_id": image_nbr,
                                         "bbox": [x, y, width, height],
                                         "area": width * height,
                                         "iscrowd": 0,
                                         "ignore": 0,
                                         "id": annotation_id,
                                         "segmentation": [[x, y, x, y + height, x + width, y + height, x + width, y]],
                                         "category_id": category_id})
        annotation_id += 1
    cv2.imwrite(f"{folder}{image_nbr:06d}.png", np.asarray(bg_image))


def calculate_overlap_percentage(x1, y1, w1, h1, x2, y2, w2, h2):
    # Calculate the coordinates of the top left and bottom right corners
    top_left_x1, top_left_y1 = x1, y1
    bottom_right_x1, bottom_right_y1 = x1 + w1, y1 + h1

    top_left_x2, top_left_y2 = x2, y2
    bottom_right_x2, bottom_right_y2 = x2 + w2, y2 + h2

    # Calculate the area of each image
    area1 = w1 * h1
    area2 = w2 * h2

    # Calculate the coordinates of the overlapping region
    top_left_x = max(top_left_x1, top_left_x2)
    top_left_y = max(top_left_y1, top_left_y2)
    bottom_right_x = min(bottom_right_x1, bottom_right_x2)
    bottom_right_y = min(bottom_right_y1, bottom_right_y2)

    # Calculate the width and height of the overlapping region
    overlapping_width = max(0, bottom_right_x - top_left_x)
    overlapping_height = max(0, bottom_right_y - top_left_y)

    # Calculate the area of the overlapping region
    overlapping_area = overlapping_width * overlapping_height

    # Calculate the percentage of overlap
    overlap_percentage = (overlapping_area / min(area1, area2)) * 100

    return overlap_percentage


def get_image():
    image_choice = random.randint(0, len(image_files) - 1)
    imageread = cv2.imread(image_files[image_choice], cv2.IMREAD_UNCHANGED)
    # imageread = cv2.cvtColor(imageread, cv2.COLOR_BGRA2RGBA)
    width = imageread.shape[1]
    height = imageread.shape[0]
    hv = int(0.3 * height)
    wv = int(0.3 * width)
    # specifying the points in the source image which is to be transformed
    # to the corresponding points in the destination image
    locks = random.sample([1, 2, 3, 4], 2)
    if 1 in locks:
        pt1 = [0, 0]
    else:
        pt1 = [0 + randint(-wv, 0), 0 + randint(-hv, 0)]
    if 2 in locks:
        pt2 = [0, height]
    else:
        pt2 = [0 + randint(-wv, 0), height + randint(0, hv)]
    if 3 in locks:
        pt3 = [width, height]
    else:
        pt3 = [width + randint(0, wv), height + randint(0, hv)]
    if 4 in locks:
        pt4 = [width, 0]
    else:
        pt4 = [width + randint(0, wv), 0 + randint(-hv, 0)]

    points1 = np.float32([pt1, pt2, pt3, pt4])
    points2 = np.float32([[0, 0], [0, height], [width, height], [width, 0]])
    # applying getPerspectiveTransform() function to transform the perspective of the given source image to the
    # corresponding points in the destination image
    resultimage = cv2.getPerspectiveTransform(points1, points2)
    # applying warpPerspective() function to fit the size of the resulting image from getPerspectiveTransform()
    # function to the size of source image
    finalimage = cv2.warpPerspective(imageread, resultimage, (width, height))
    if bool(random.getrandbits(1)):
        number_square = random.randint(1, 4)
        for i in range(number_square):
            size_square = random.uniform(0.2, 0.5)
            start_x = random.randint(0, int((1 - size_square) * width))
            start_y = random.randint(0, int((1 - size_square) * height))
            color_b = random.randint(0, 255)
            color_g = random.randint(0, 255)
            color_r = random.randint(0, 255)
            finalimage = cv2.rectangle(finalimage, (start_x, start_y),
                                       (int(start_x + size_square * width), int(start_y + size_square * height)),
                                       (color_b, color_g, color_r, 255), -1)
    if bool(random.getrandbits(1)):
        finalimage = ndimage.rotate(finalimage, random.randint(0, 360))
    return finalimage, image_choice + 1


def build_coco_json():
    coco_json = {"categories": [], "images": [], "annotations": []}
    for i in range(len(image_files)):
        coco_json["categories"].append({"supercategory": "none", "id": i+1, "name": pathlib.Path(image_files[i]).stem})
    return coco_json


total_number_generated_images = 10
coco_train = build_coco_json()
annotation_id = 1
image_id = 1
os.mkdir("./distorted/train")
os.mkdir("./distorted/test")
for i in range(round(0.8 * total_number_generated_images)):
    get_full_image("./distorted/train/", image_id, coco_train)
    image_id += 1
with open("./distorted/train.json", 'w') as outfile:
    json.dump(coco_train, outfile)
coco_test = build_coco_json()
for i in range(round(0.2 * total_number_generated_images)):
    get_full_image("./distorted/test/", image_id, coco_test)
    image_id += 1
with open("./distorted/test.json", 'w') as outfile:
    json.dump(coco_test, outfile)
