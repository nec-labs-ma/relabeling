import torch
import pickle
from torchvision import models, transforms
from PIL import Image
from glob import glob
import json
import os
import cv2
import numpy as np

# Paths
model_path = "resnet101_mapillary_vehicles_classifier.pth"
class_map_path = "mapillary_vehicles_class_to_idx.pkl"

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
    correct_vlm1=0
    total=0
    files = glob('/net/acadia10a/data/user/mapillary/mapillary-2.0/validation/v2.0/polygons/*.json')
    for poly_file in files:
        all_polygon_points = read_polygon(poly_file)
        filename = os.path.basename(poly_file).replace('json', 'jpg')
        image_path = '/net/acadia10a/data/user/mapillary/mapillary-2.0/validation/images/%s'%filename
        image = cv2.imread(image_path)
        for key, polygon_points in all_polygon_points.items():
            for poly in polygon_points:
                poly = np.array(poly).astype(np.int32)
                #cv2.polylines(image, [poly], isClosed=True, color=(0, 255, 0), thickness=5)
                x, y, w, h = cv2.boundingRect(poly)
                #cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 5)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                cropped_image = image[y:y + h, x:x + w] 
                cropped_image = upscale_and_resize(cropped_image, 28)
                dino_response = predict_image_label(cropped_image)
                gt = None
                if 'vehicle' in key:
                    gt_list = key.split('--')
                    if len(gt_list)==3:
                        gt = key.split('--')[2]
                    else:
                        gt = key
                gt = gt.lower()
                if gt in ["car", "bus", "truck", "bicycle", "motorcycle", "boat", "on-rails"] and dino_response:
                    response11 = dino_response.lower()
                    if response11 == 'suv' or response11 == 'van':
                        response11 = 'car'
                    if response11 == 'train':
                        response11 = 'on-rails'
                    print("Prediction1:", response11, " gt:", gt)
                    if gt == response11:
                        correct_vlm1+=1
                    total+=1
                elif dino_response == None:
                    correct_vlm1+=1
                    total+=1

    save_path = 'classifier'
    with open('%s_mapillary_vehicles.txt'%save_path, 'w') as f:
        f.write(str(correct_vlm1/total))
        f.write('\n')

if __name__ == "__main__":
    main()