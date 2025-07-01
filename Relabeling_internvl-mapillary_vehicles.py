#!/usr/bin/env python
# coding: utf-8

# ## Loading the dataset

# ### List all the categories available

# In[36]:


### read the json file
import json
import argparse
import os
from glob import glob
import sys
import re
cats = json.load(open('/net/acadia10a/data/user/mapillary/mapillary-2.0/config_v2.0.json', 'r'))
cats = {i['readable']: i['color'] for i in cats['labels']}


# ### Fetch vehicles and visualize the objects

# In[35]:
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

### read the polygon file
import cv2
import numpy as np
from PIL import Image
vehicles_categories = {
  "Bicycle": [119, 11, 32],
  "Boat": [150, 0, 255],
  "Bus": [0, 60, 100],
  "Car": [0, 0, 142],
  "Caravan": [0, 0, 90],
  "Motorcycle": [0, 0, 230],
  "On Rails": [0, 80, 100],
  "Other Vehicle": [128, 64, 64],
  "Trailer": [0, 0, 110],
  "Truck": [0, 0, 70],
  "Vehicle Group": [0, 0, 142],
  "Wheeled Slow": [0, 0, 192],
  "Car Mount": [32, 32, 32]
}
cat = 'vehicle'
def read_polygon(polygon_path, cat):
    polys = json.load(open(polygon_path, 'r'))
    vehicle_instances = {}
    for i in polys['objects']:
        if cat in i['label']:
            if i['label'] not in vehicle_instances:
                vehicle_instances[i['label']] = [i['polygon']]
            else:
                vehicle_instances[i['label']].append(i['polygon'])
    return vehicle_instances

def visualization(image_path, all_polygon_points):
    image = cv2.imread(image_path)

    # Draw polygon
    for key, polygon_points in all_polygon_points.items():
        for poly in polygon_points:
            poly = np.array(poly).astype(np.int32)
            cv2.polylines(image, [poly], isClosed=True, color=(0, 255, 0), thickness=5)
            x, y, w, h = cv2.boundingRect(poly)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, key, (x, y - 10), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
    return Image.fromarray(image)


# ### Load InternVL to relabel the vehicles into following categories: 
# Car
# 
# Bus
# 
# Truck
# 
# Bicycle
# 
# Motorcycle
# 
# Boat
# 
# Trailer
# 
# Caravan
# 
# On Rails
# 
# Other Vehicle
# 
# Vehicle Group
# 
# Wheeled Slow

# In[34]:


from PIL import Image, ImageDraw
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import numpy as np
import torch
import torchvision.transforms as T

from segment_anything import sam_model_registry, SamPredictor
model_type="vit_h"
sam_checkpoint="sam_vit_h.pth"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
device = "cuda" if torch.cuda.is_available() else "cpu"
sam.to(device)
predictor = SamPredictor(sam)

def get_sam_cutout_from_bbox(image_path, bbox):
    """
    Run SAM on an image and a bounding box to get a mask cutout.

    Args:
        image_path (str): Path to the image file.
        bbox (tuple): Bounding box as (x_min, y_min, x_max, y_max).
        sam_checkpoint (str): Path to the SAM model checkpoint.
        model_type (str): Type of the SAM model ('vit_h', 'vit_l', 'vit_b').

    Returns:
        np.ndarray: Mask cutout image (same shape as image, but only segmented area retained).
        np.ndarray: Boolean segmentation mask.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Set image
    predictor.set_image(image_rgb)

    # Format box as numpy array
    input_box = np.array(bbox)

    # Predict with SAM
    masks, scores, _ = predictor.predict(box=input_box[None, :], multimask_output=False)

    mask = masks[0]  # (H, W) boolean array

    # Apply mask to original image (cutout)
    cutout = image_rgb.copy()
    cutout[~mask] = 0  # set background to black

    return cutout, mask


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image, input_size=448, max_num=12):
    #image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def remove_special_characters(text):
    # Remove all characters except letters, numbers, and spaces
    text = text.lower()
    return re.sub(r'[^A-Za-z0-9\s]', '', text)

def main():
    # parser = argparse.ArgumentParser(description="Process a batch of subdirectories to generate 3D-to-2D annotations.")
    # parser.add_argument('--subdirs', nargs='+', required=True, help='List of subdirectories to process')
    # args = parser.parse_args()
    # path = args.subdirs[0]
    # path = path.replace('[', '').replace(']', '').replace("'", "")
    path = sys.argv[1]
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True).eval().cuda()
    tokenizer1 = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    prompt1 = """<image>\nYou are an expert in autonomous driving, specializing in analyzing traffic scenes.                
            Your task is to look at the obstacle in the image and output the response in the format below. think step by step before deciding the answer and return only ONE word answer
            Strictly follow the rules.
            {
                [Choose ONE category among the following for the most dominant object in the bounding box: "car", "bus", "SUV", "truck", "bicycle", "motorcycle", "boat", "trailer", "train"]                
            }"""

    prompt2 = """<image>\nYou are an expert in autonomous driving, specializing in analyzing traffic scenes.                
                Your task is to look at the obstacle in the red bbox  and output the response in the format below. think step by step before deciding the answer and return only ONE word answer
                Strictly follow the rules.
                {
                    [Choose ONE category among the following for the most dominant object in the bounding box: "car", "bus", "SUV", "truck", "bicycle", "motorcycle", "boat", "trailer", "train"]                
                }"""


    generation_config = dict(max_new_tokens=1024, do_sample=True)
    correct_vlm1=0
    correct_vlm2=0
    correct_vlm3=0
    total=0
    results = {}

    files = glob('/net/acadia10a/data/user/mapillary/mapillary-2.0/validation/v2.0/polygons/*.json')
    for poly_file in files:
        all_polygon_points = read_polygon(poly_file, 'vehicle')
        filename = os.path.basename(poly_file).replace('json', 'jpg')
        image_path = '/net/acadia10a/data/user/mapillary/mapillary-2.0/validation/images/%s'%filename
        for key, polygon_points in all_polygon_points.items():
            for poly in polygon_points:
                poly = np.array(poly).astype(np.int32)
                image = cv2.imread(image_path)
                #cv2.polylines(image, [poly], isClosed=True, color=(0, 255, 0), thickness=5)
                x, y, w, h = cv2.boundingRect(poly)
                x1, y1, x2, y2 = x, y, x+w, y+h
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                cropped_image = image[y:y + h, x:x + w] 
                pixel_values = load_image(Image.fromarray(cropped_image), max_num=12).to(torch.bfloat16).cuda()
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 5)
                pixel_values_bigg = load_image(Image.fromarray(image), max_num=12).to(torch.bfloat16).cuda()
                sam_cutout = get_sam_cutout_from_bbox(image_path, [x1, y1, x2, y2])[0]
                cropped_sam = Image.fromarray(sam_cutout[y1:y2, x1:x2])
                pixel_values_sam = load_image(cropped_sam, max_num=12).to(torch.bfloat16).cuda()
                response11 = model.chat(tokenizer1, pixel_values, prompt1, generation_config)
                response12 = model.chat(tokenizer1, pixel_values_bigg, prompt2, generation_config)
                response13 = model.chat(tokenizer1, pixel_values_sam, prompt1, generation_config)
                response11 = remove_special_characters(response11).replace('category', '').replace('json', '').replace('\n', '').replace(' ', '').replace('object', '').replace('categorie', '')
                response12 = remove_special_characters(response12).replace('category', '').replace('json', '').replace('\n', '').replace(' ', '').replace('object', '').replace('categorie', '')
                response13 = remove_special_characters(response13).replace('category', '').replace('json', '').replace('\n', '').replace(' ', '').replace('object', '').replace('categorie', '')
                if cat in key:
                    gt_list = key.split('--')
                if len(gt_list)==3:
                    gt = key.split('--')[2]
                else:
                    gt = key
                if gt != 'vehicle-group':
                    response11 = response11.lower()
                    response12 = response12.lower()
                    if response11 == 'suv':
                        response11 = 'car'
                    if response12 == 'suv':
                        response12 = 'car'
                    if response13 == 'suv':
                        response13 = 'car'
                    if response11 == 'train':
                        response11 = 'on-rails'
                    if response12 == 'train':
                        response12 = 'on-rails'
                    if response13 == 'train':
                        response13 = 'on-rails'
    #                 response12 = vlm_response2.lower()
                    print("Prediction1:", response11, " gt:", gt, "Prediction2:", response12, response13)
                    if image_path in results:
                        results[image_path].append([gt, response11, response12, response13])
                    else:
                        results[image_path] = [[gt, response11, response12, response13]]
                    total+=1
    save_path = path.replace('/', '_')
    with open('%s_vehicles_mapillary.json'%save_path, 'w') as f:
        json.dump(results, f, indent=2)

    with open('%s_vehicles_mapillary.txt'%save_path, 'w') as f:
        f.write(str(correct_vlm1/total))
        f.write('\n')
        f.write(str(correct_vlm2/total))
        f.write('\n')
        f.write(str(correct_vlm3/total))

                
if __name__ == "__main__":
    main()




