import torch
import pickle
from torchvision import models, transforms
from PIL import Image
from glob import glob
import json
import os
import cv2
import numpy as np
import math

# Paths
model_path = "resnet101_signs_classifier.pth"
class_map_path = "signs_class_to_idx.pkl"

# Load model and class mapping
checkpoint = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
class_to_idx = checkpoint['class_to_idx']
idx_to_class = {v: k for k, v in class_to_idx.items()}

model = models.resnet101(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_to_idx))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Inference function
def predict_image_label(pil_image):
    image = pil_image
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        predicted_idx = output.argmax(dim=1).item()
        return idx_to_class[predicted_idx]

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

def read_polygon(polygon_path):
    polys = json.load(open(polygon_path, 'r'))
    vehicle_instances = {}
    for i in polys['objects']:
        if 'vehicle' in i['label']:
            if i['label'] not in vehicle_instances:
                vehicle_instances[i['label']] = [i['polygon']]
            else:
                vehicle_instances[i['label']].append(i['polygon'])
    return vehicle_instances

def main():
    data = json.load(open('instances_default.json', 'r'))
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    correct_vlm1=0
    total=0
    img_count=0
    os.system('rm -rf wrong/*')
    for poly_file in data['annotations']:
        filename = data['images'][poly_file['image_id']-1]['file_name']
        ground_truth = poly_file['attributes']['Type of Signs'].replace('- ', '')
        ground_shape = data['categories'][poly_file['category_id']-1]['name']
        if ground_truth=='Others' and ground_shape == 'Sign':
            continue
        image_path = '/net/acadia10a/data/user/mapillary/mapillary-2.0/validation/images/%s'%filename
        image = cv2.imread(image_path)
        #cv2.polylines(image, [poly], isClosed=True, color=(0, 255, 0), thickness=5)
        x, y, w, h = poly_file['bbox']
        x = math.floor(x)
        y = math.floor(y)
        w = math.ceil(w)
        h = math.ceil(h)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cropped_image = image[y:y + h, x:x + w]
        cropped_image = upscale_and_resize(cropped_image, 28)
        #print(ground_truth)
        dino_response = predict_image_label(cropped_image) 
        gt = ground_truth.lower()
        print("Prediction1:", dino_response, " gt:", gt)
        img_count+=1
        if dino_response:
            if dino_response.lower() == gt:
                correct_vlm1+=1
        else:
            if gt == 'others':
                correct_vlm1+=1
        total+=1

    save_path = 'classifier'
    with open('%s_signs.txt'%save_path, 'w') as f:
        f.write(str(correct_vlm1/total))
        f.write('\n')

if __name__ == "__main__":
    main()
