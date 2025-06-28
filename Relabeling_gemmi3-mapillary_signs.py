#!/usr/bin/env python
# coding: utf-8

# ## Loading the dataset

# ### List all the categories available

# In[36]:


### read the json file
import json
import os
import re
from glob import glob
from accelerate import infer_auto_device_map
import math
import sys

cats = json.load(open('/net/acadia10a/data/sparsh/mapillary/mapillary-2.0/config_v2.0.json', 'r'))
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


import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import os
from glob import glob
import json
import pickle
import re
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
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

def upscale_and_resize(image, factor):
    if image is None or image.size == 0:
        return None
    height, width = image.shape[:2]

    # Upscale image if necessary
    if height <= factor or width <= factor:
        new_height = max(factor + 1, height * 2)
        new_width = max(factor + 1, width * 2)
        image = cv2.resize(image, (new_width, new_height))

    return Image.fromarray(image)

def return_label_vlm(pil_image, text, processor, model):
    inputs = processor(
    text=[text],
    images=pil_image,
    videos=None,
    padding=True,
    return_tensors="pt",
)
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text

def remove_special_chars(text):
    text = text.lower()
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)
def main():
    path = sys.argv[1]

    model = Gemma3ForConditionalGeneration.from_pretrained(
        path, device_map="auto"
    ).eval()

    processor = AutoProcessor.from_pretrained(path)

    prompt1 = """<image>\nYou are an expert in autonomous driving, specializing in analyzing traffic scenes.                
                Your task is to look at the traffic signs in the image and output the response in the format below. think step by step before deciding the answer and return only ONE word answer
                Strictly follow the rules.
                {
                    [Choose ONE category among the following for the most dominant object in the bounding box: "stop signs", "speed limit signs", "yield signs", "do not enter", "crosswalk signs", "parking signs", 'roundabout signs", "turn signs", "cycle lane signs", "no parking", "others"]                
                }"""

    prompt2 = """<image>\nYou are an expert in autonomous driving, specializing in analyzing traffic scenes.                
                Your task is to look at the  traffic signs in the red bbox  and output the response in the format below. think step by step before deciding the answer and return only ONE word answer
                Strictly follow the rules.
                {
                    [Choose ONE category among the following for the most dominant object in the bounding box: "stop signs", "speed limit signs", "yield signs", "do not enter", "crosswalk signs", "parking signs", 'roundabout signs", "turn signs", "cycle lane signs", "no parking", "others"] 
                }"""

    messages1 = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": prompt1},
            ],
        }
    ]

    messages2 = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": prompt2},
            ],
        }
    ]

    text1 = processor.apply_chat_template(
        messages1, tokenize=False, add_generation_prompt=True
    )

    text2 = processor.apply_chat_template(
        messages2, tokenize=False, add_generation_prompt=True
    )

    data = json.load(open('instances_default.json', 'r'))
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    correct_vlm1=0
    correct_vlm2=0
    correct_vlm3=0
    results = {}
    total=0


    for poly_file in data['annotations']:
        filename = data['images'][poly_file['image_id']-1]['file_name']
        ground_truth = poly_file['attributes']['Type of Signs'].replace('- ', '')
        ground_shape = data['categories'][poly_file['category_id']-1]['name']
        if ground_truth=='Others' and ground_shape == 'Sign':
            continue
        image_path = '/net/acadia10a/data/sparsh/mapillary/mapillary-2.0/validation/images/%s'%filename
        print(ground_truth, ground_shape)
        image = cv2.imread(image_path)
        #cv2.polylines(image, [poly], isClosed=True, color=(0, 255, 0), thickness=5)
        x, y, w, h = poly_file['bbox']
        x = math.floor(x)
        y = math.floor(y)
        w = math.ceil(w)
        h = math.ceil(h)
        x1, y1, x2, y2 = x, y, x+w, y+h
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cropped_image = image[y:y + h, x:x + w] 
        cropped_image = upscale_and_resize(cropped_image, 28)
        vlm_response1 = return_label_vlm(cropped_image, text1, processor, model)[0].replace(' ', '').replace('\n', '')
        vlm_response1 = remove_special_chars(vlm_response1)
        vlm_response1 = vlm_response1.replace('category', '').replace('json', '').replace('\n', '').replace(' ', '').replace('object', '').replace('categorie', '')
        vlm_response2 = return_label_vlm(Image.fromarray(image), text2, processor, model)[0].replace(' ', '').replace('\n', '')
        vlm_response2 = remove_special_chars(vlm_response2)
        vlm_response2 = vlm_response2.replace('category', '').replace('json', '').replace('\n', '').replace(' ', '').replace('object', '').replace('categorie', '')
        sam_cutout = get_sam_cutout_from_bbox(image_path, [x1, y1, x2, y2])[0]
        cropped_sam = Image.fromarray(sam_cutout[y1:y2, x1:x2])
        vlm_response3 = return_label_vlm(cropped_sam, text1, processor, model)[0].replace(' ', '').replace('\n', '')
        vlm_response3 = remove_special_chars(vlm_response3)
        vlm_response3 = vlm_response3.replace('category', '').replace('json', '').replace('\n', '').replace(' ', '').replace('object', '').replace('categorie', '')
        gt=ground_truth
        gt = gt.lower()
        gt = gt.replace(' ', '')
        response11 = vlm_response1.lower()
        response12 = vlm_response2.lower()
        response13 = vlm_response3.lower()
        if response11 == 'speedlimitsigns':
            response11 = 'speedsigns'
        if response12 == 'speedlimitsigns':
            response12 = 'speedsigns'
        if response13 == 'speedlimitsigns':
            response13 = 'speedsigns'
        if response11 == 'roundaboutsigns':
            response11 = 'roundabout'
        if response12 == 'roundaboutsigns':
            response12 = 'roundabout'
        if response13 == 'roundaboutsigns':
            response13 = 'roundabout'
        if response11 == 'parking':
            response11 = 'parkingsigns'
        if response12 == 'parking':
            response12 = 'parkingsigns'
        if response13 == 'parking':
            response13 = 'parkingsigns'
        print("Prediction1:", response11, " gt:", gt, "Prediction2:", response12, "Prediction3:", response13)
        if gt == response11:
            correct_vlm1+=1
        if gt == response12:
            correct_vlm2+=1
        if gt == response13:
            correct_vlm3+=1
        total+=1
        if image_path in results:
            results[image_path].append([gt, response11, response12, response13])
        else:
            results[image_path] = [[gt, response11, response12, response13]]
    save_path = path.replace('/', '_')
    with open('%s_signs_mapillary.json'%save_path, 'w') as f:
        json.dump(results, f, indent=2)

    with open('%s_signs_mapillary.txt'%save_path, 'w') as f:
        f.write(str(correct_vlm1/total))
        f.write('\n')
        f.write(str(correct_vlm2/total))
        f.write('\n')
        f.write(str(correct_vlm3/total))

                
if __name__ == "__main__":
    main()
