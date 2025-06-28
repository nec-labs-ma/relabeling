import cv2
import numpy as np
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import os
from glob import glob
import json
import pickle
from PIL import Image, ImageDraw, ImageFont
#from transformers import AutoImageProcessor, AutoModel, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers import AutoImageProcessor, AutoModel, AutoProcessor
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# class_files = {
#     # -- class 1
#     "car": ["/net/acadia14a/data/sparsh/Relabeling/dinov2_data/car/facebook_dinov2-giant_car.pt"],
#     "bus": ["/net/acadia14a/data/sparsh/Relabeling/dinov2_data/bus/facebook_dinov2-giant_bus.pt"],
#     "train":['/net/acadia14a/data/sparsh/Relabeling/dinov2_data/train/facebook_dinov2-giant_train.pt'],
#     'truck': ['/net/acadia14a/data/sparsh/Relabeling/dinov2_data/truck/facebook_dinov2-giant_truck.pt'],
#     "suv": ['/net/acadia14a/data/sparsh/Relabeling/dinov2_data/suv/facebook_dinov2-giant_suv.pt'],
#     "motorcycle":['/net/acadia14a/data/sparsh/Relabeling/dinov2_data/motorcycle/facebook_dinov2-giant_motorcycle.pt'],
#     'bicycle': ['/net/acadia14a/data/sparsh/Relabeling/dinov2_data/bicycle/facebook_dinov2-giant_bicycle.pt'],
#     "boat": ['/net/acadia14a/data/sparsh/Relabeling/dinov2_data/boat/facebook_dinov2-giant_boat.pt'],
#     "van":['/net/acadia14a/data/sparsh/Relabeling/dinov2_data/van/facebook_dinov2-giant_van.pt'],
#     "person": ['/net/acadia14a/data/sparsh/Relabeling/dinov2_data/CyclistDetectorTrainingDatav3coco/train/cropped_images/person/facebook_dinov2-giant_person.pt'],
#     "bicyclist": ['/net/acadia14a/data/sparsh/Relabeling/dinov2_data/CyclistDetectorTrainingDatav3coco/train/cropped_images/bicyclist/facebook_dinov2-giant_bicyclist.pt'],
#     "motorcyclist": ['/net/acadia14a/data/sparsh/Relabeling/dinov2_data/CyclistDetectorTrainingDatav3coco/train/cropped_images/motorcyclist/facebook_dinov2-giant_motorcyclist.pt']
# }

class_files = {
    "person": ['/net/acadia14a/data/sparsh/Relabeling/dinov2_data/humans/CyclistDetectorTrainingDatav3coco/train/cropped_images/person/facebook_dinov2-giant_person.pt', '/net/acadia14a/data/sparsh/Relabeling/dinov2_data/humans/Person_v9-6040_coco/train/cropped_images/Person/facebook_dinov2-giant_Person.pt', '/net/acadia14a/data/sparsh/Relabeling/dinov2_data/humans/Person_v9-6040_coco/train/cropped_images/person/facebook_dinov2-giant_person.pt'],
    "bicyclist": ['/net/acadia14a/data/sparsh/Relabeling/dinov2_data/humans/CyclistDetectorTrainingDatav3coco/train/cropped_images/bicyclist/facebook_dinov2-giant_bicyclist.pt', '/net/acadia14a/data/sparsh/Relabeling/dinov2_data/humans/bike_detect/train/cropped_images/cyclist/facebook_dinov2-giant_cyclist.pt'],
    "motorcyclist": ['/net/acadia14a/data/sparsh/Relabeling/dinov2_data/humans/CyclistDetectorTrainingDatav3coco/train/cropped_images/motorcyclist/facebook_dinov2-giant_motorcyclist.pt', '/net/acadia14a/data/sparsh/Relabeling/dinov2_data/humans/motorbike/train/cropped_images/motorcyclist/facebook_dinov2-giant_motorcyclist.pt']
}

class_features = {}

for class_name, pt_files in class_files.items():
    # Load all features for the current class and store them in a list
    loaded_features = [torch.load(pt_file) for pt_file in pt_files]

    # Concatenate all loaded features along the batch dimension
    if loaded_features:  # Check if there are any loaded features
        stacked_features = torch.cat(loaded_features, dim=0)
        class_features[class_name] = stacked_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load DINOv2 model and image processor
model_name = 'facebook/dinov2-giant'
model = AutoModel.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)
model.eval()

model = model.to(device)


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
        if 'human' in i['label']:
            if i['label'] not in vehicle_instances:
                vehicle_instances[i['label']] = [i['polygon']]
            else:
                vehicle_instances[i['label']].append(i['polygon'])
    return vehicle_instances

def extract_features(image):
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        features = outputs.last_hidden_state

    return features.mean(dim=1)



def return_label(pil_image, class_features, top_two=False):
    feature_tensors = torch.stack([features.mean(dim=0) for features in class_features.values()]).to(device)
    labels = list(class_features.keys())
    try:
        proposal_features = extract_features(np.array(pil_image)).reshape(1, -1)
    except ZeroDivisionError as e:
        print(f"Error: {e}")
        print(f"Shape of proposal when error occurred: {np.array(pil_image).shape}")
        # raise
        return None
    similarities = torch.nn.functional.cosine_similarity(proposal_features, feature_tensors, dim=1)
    normalized_similarities = (similarities + 1) / 2

    valid_indices = (normalized_similarities >= 0.8).nonzero(as_tuple=True)[0]
    valid_similarities = normalized_similarities[valid_indices]
    if valid_similarities.size(0) > 0:
        if top_two:
            _, top_indices = torch.topk(valid_similarities, min(2, valid_similarities.size(0)))
            top_classes = [(labels[valid_indices[idx]], valid_similarities[idx].item()) for idx in top_indices]
            top_classes = sorted(top_classes, key=lambda x: x[0])
        else:
            highest_similarity, idx = torch.max(valid_similarities, 0)
            best_class = labels[valid_indices[idx]]
            score = highest_similarity.item()
    else:
        return None
    if top_two and top_classes:
        for j, (class_name, similarity) in enumerate(top_classes):
            usdot_class = class_name
            label = f'{usdot_class}: {similarity:.2f}'
    else:
        usdot_class = best_class
        label = f'{usdot_class}: {score:.2f}'
    return usdot_class


def main():
    correct_vlm1=0
    total=0
    files = glob('/net/acadia10a/data/sparsh/mapillary/mapillary-2.0/validation/v2.0/polygons/*.json')
    for poly_file in files:
        all_polygon_points = read_polygon(poly_file)
        filename = os.path.basename(poly_file).replace('json', 'jpg')
        image_path = '/net/acadia10a/data/sparsh/mapillary/mapillary-2.0/validation/images/%s'%filename
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
                dino_response = return_label(cropped_image, class_features)
                gt = None
                if 'human' in key:
                    gt_list = key.split('--')
                    if len(gt_list)==3:
                        gt = key.split('--')[2]
                    else:
                        gt = key
                if gt in ['individual', 'bicyclist', 'motorcyclist'] and gt != 'person-group' and dino_response:
                    response11 = dino_response.lower()
                    if response11 == 'person':
                        response11 = 'individual'
                    print("Prediction1:", response11, " gt:", gt)
                    if gt == response11:
                        correct_vlm1+=1
                    total+=1
                elif dino_response == None:
                    correct_vlm1+=1
                    total+=1

    save_path = 'dino_v2'
    with open('%s_pedestrains.txt'%save_path, 'w') as f:
        f.write(str(correct_vlm1/total))
        f.write('\n')

if __name__ == "__main__":
    main()
