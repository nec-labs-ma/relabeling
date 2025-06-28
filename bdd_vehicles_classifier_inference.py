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
model_path = "resnet101_bdd_vehicles_classifier.pth"
class_map_path = "bdd_vehicles_class_to_idx.pkl"

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

def main():
    correct_vlm1=0
    correct_vlm2=0
    total=0

    cats = json.load(open('/net/acadia7a/data/samuel/bdd100k/det_val.json', 'r'))


    valid_cats = ['car', 'motorcycle', 'bus', 'truck', 'bicycle']
    for i in cats:
        cat_image = i['labels']
        image_path = '/net/acadia7a/data/samuel/bdd100k/images/100k/val/%s'%i['name']
        image = cv2.imread(image_path)
        for j in cat_image:
            if j['category'] in valid_cats:
                if 'box2d' in j:
                    x1, y1, x2, y2 = int(j['box2d']['x1']), int(j['box2d']['y1']), int(j['box2d']['x2']), int(j['box2d']['y2'])
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    cropped_image = image[y1:y2, x1:x2]
                    cropped_image = upscale_and_resize(cropped_image, 28)
                    dino_response = predict_image_label(cropped_image)
                    gt=j['category']
                    if gt != 'vehicle-group' and dino_response:
                        response11 = dino_response.lower()
                        if response11 == 'suv' or response11 == 'van':
                            response11 = 'car'
                        if response11 == 'train':
                            response11 = 'on-rails'
                        print("Prediction1:", response11, " gt:", gt)
                        if gt == response11:
                            correct_vlm1+=1
                        total+=1
    save_path = 'classifier'
    with open('%s_bdd_vehicles.txt'%save_path, 'w') as f:
        f.write(str(correct_vlm1/total))
        f.write('\n')

if __name__ == "__main__":
    main()