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
import math


class_files = {'Others': ['/net/acadia14a/data/user/Relabeling/dinov2_data/TrafficRoadSignscoco/train/cropped_images/Others/facebook_dinov2-giant_Others.pt'],
 'Do not enter': ['/net/acadia14a/data/user/Relabeling/dinov2_data/TrafficRoadSignscoco/train/cropped_images/Do_not_enter/facebook_dinov2-giant_Do_not_enter.pt'],
 'Roundabout': ['/net/acadia14a/data/user/Relabeling/dinov2_data/TrafficRoadSignscoco/train/cropped_images/Roundabout/facebook_dinov2-giant_Roundabout.pt', '/net/acadia14a/data/user/Relabeling/dinov2_data/European road signs_v8i_coco/train/cropped_images/roundabout/facebook_dinov2-giant_roundabout.pt', '/net/acadia14a/data/user/Relabeling/dinov2_data/Traffic Signs Detection Europe_v14-traffic-signs-detection-4_5k-images_coco/train/cropped_images/mand_roundabout/facebook_dinov2-giant_mand_roundabout.pt', '/net/acadia14a/data/user/Relabeling/dinov2_data/Traffic Signs Detection Europe_v14-traffic-signs-detection-4_5k-images_coco/train/cropped_images/warn_roundabout/facebook_dinov2-giant_warn_roundabout.pt'],
 'Yield signs': ['/net/acadia14a/data/user/Relabeling/dinov2_data/TrafficRoadSignscoco/train/cropped_images/Yield_signs/facebook_dinov2-giant_Yield_signs.pt'],
 'Turn signs': ['/net/acadia14a/data/user/Relabeling/dinov2_data/TrafficRoadSignscoco/train/cropped_images/Turn_signs/facebook_dinov2-giant_Turn_signs.pt'],
 'Cycle Lane Signs': ['/net/acadia14a/data/user/Relabeling/dinov2_data/cycle_lane_signs/facebook_dinov2-giant_cycle_lane_signs.pt', '/net/acadia14a/data/user/Relabeling/dinov2_data/Traffic Signs Detection Europe_v14-traffic-signs-detection-4_5k-images_coco/train/cropped_images/mand_bike_lane/facebook_dinov2-giant_mand_bike_lane.pt', '/net/acadia14a/data/user/Relabeling/dinov2_data/Traffic Signs Detection Europe_v14-traffic-signs-detection-4_5k-images_coco/train/cropped_images/warn_cyclists/facebook_dinov2-giant_warn_cyclists.pt'],
 'Speed Signs': ['/net/acadia14a/data/user/Relabeling/dinov2_data/TrafficRoadSignscoco/train/cropped_images/Speed_Signs/facebook_dinov2-giant_Speed_Signs.pt'],
 'Stop signs': ['/net/acadia14a/data/user/Relabeling/dinov2_data/TrafficRoadSignscoco/train/cropped_images/Stop_signs/facebook_dinov2-giant_Stop_signs.pt'],
 'Crosswalk Signs': ['/net/acadia14a/data/user/Relabeling/dinov2_data/TrafficRoadSignscoco/train/cropped_images/Crosswalk_Signs/facebook_dinov2-giant_Crosswalk_Signs.pt', '/net/acadia14a/data/user/Relabeling/dinov2_data/European road signs_v8i_coco/train/cropped_images/Pedestrian crossing/facebook_dinov2-giant_Pedestrian_crossing.pt', '/net/acadia14a/data/user/Relabeling/dinov2_data/European road signs_v8i_coco/train/cropped_images/Pedestrian crossing ahead/facebook_dinov2-giant_Pedestrian_crossing_ahead.pt', '/net/acadia14a/data/user/Relabeling/dinov2_data/Traffic Signs Detection Europe_v14-traffic-signs-detection-4_5k-images_coco/train/cropped_images/info_crosswalk/facebook_dinov2-giant_info_crosswalk.pt', '/net/acadia14a/data/user/Relabeling/dinov2_data/Traffic Signs Detection Europe_v14-traffic-signs-detection-4_5k-images_coco/train/cropped_images/warn_crosswalk/facebook_dinov2-giant_warn_crosswalk.pt'],
 'Parking signs': ['/net/acadia14a/data/user/Relabeling/dinov2_data/parking sign_v1i_coco/train/cropped_images/Parking-Sign/facebook_dinov2-giant_Parking-Sign.pt', '/net/acadia14a/data/user/Relabeling/dinov2_data/parking_signs/facebook_dinov2-giant_parking_signs.pt', '/net/acadia14a/data/user/Relabeling/dinov2_data/Traffic Signs Detection Europe_v14-traffic-signs-detection-4_5k-images_coco/train/cropped_images/info_parking/facebook_dinov2-giant_info_parking.pt'],
 'No parking': ['/net/acadia14a/data/user/Relabeling/dinov2_data/traffic_signs/train/cropped_images/noparking/facebook_dinov2-giant_noparking.pt', '/net/acadia14a/data/user/Relabeling/dinov2_data/European road signs_v8i_coco/train/cropped_images/No parking/facebook_dinov2-giant_No_parking.pt', '/net/acadia14a/data/user/Relabeling/dinov2_data/Traffic Signs Detection Europe_v14-traffic-signs-detection-4_5k-images_coco/train/cropped_images/mand_roundabout/facebook_dinov2-giant_mand_roundabout.pt', '/net/acadia14a/data/user/Relabeling/dinov2_data/Traffic Signs Detection Europe_v14-traffic-signs-detection-4_5k-images_coco/train/cropped_images/forb_stopping/facebook_dinov2-giant_forb_stopping.pt']}


# class_files = {
#  'Do not enter': ['/net/acadia14a/data/user/Relabeling/dinov2_data/TrafficRoadSignscoco/train/cropped_images/Do_not_enter/facebook_dinov2-giant_Do_not_enter.pt'],
#  'Roundabout': ['/net/acadia14a/data/user/Relabeling/dinov2_data/TrafficRoadSignscoco/train/cropped_images/Roundabout/facebook_dinov2-giant_Roundabout.pt', '/net/acadia14a/data/user/Relabeling/dinov2_data/European road signs_v8i_coco/train/cropped_images/roundabout/facebook_dinov2-giant_roundabout.pt'],
#  'Yield signs': ['/net/acadia14a/data/user/Relabeling/dinov2_data/TrafficRoadSignscoco/train/cropped_images/Yield_signs/facebook_dinov2-giant_Yield_signs.pt'],
#  'Turn signs': ['/net/acadia14a/data/user/Relabeling/dinov2_data/TrafficRoadSignscoco/train/cropped_images/Turn_signs/facebook_dinov2-giant_Turn_signs.pt'],
#  'Cycle Lane Signs': ['/net/acadia14a/data/user/Relabeling/dinov2_data/cycle_lane_signs/facebook_dinov2-giant_cycle_lane_signs.pt'],
#  'Speed Signs': ['/net/acadia14a/data/user/Relabeling/dinov2_data/TrafficRoadSignscoco/train/cropped_images/Speed_Signs/facebook_dinov2-giant_Speed_Signs.pt'],
#  'Stop signs': ['/net/acadia14a/data/user/Relabeling/dinov2_data/TrafficRoadSignscoco/train/cropped_images/Stop_signs/facebook_dinov2-giant_Stop_signs.pt', '/net/acadia14a/data/user/Relabeling/dinov2_data/European road signs_v8i_coco/train/cropped_images/Stop/facebook_dinov2-giant_Stop.pt'],
#  'Crosswalk Signs': ['/net/acadia14a/data/user/Relabeling/dinov2_data/TrafficRoadSignscoco/train/cropped_images/Crosswalk_Signs/facebook_dinov2-giant_Crosswalk_Signs.pt', '/net/acadia14a/data/user/Relabeling/dinov2_data/European road signs_v8i_coco/train/cropped_images/Pedestrian crossing/facebook_dinov2-giant_Pedestrian_crossing.pt', '/net/acadia14a/data/user/Relabeling/dinov2_data/European road signs_v8i_coco/train/cropped_images/Pedestrian crossing ahead/facebook_dinov2-giant_Pedestrian_crossing_ahead.pt'],
#  'Parking signs': ['/net/acadia14a/data/user/Relabeling/dinov2_data/parking sign_v1i_coco/train/cropped_images/Parking-Sign/facebook_dinov2-giant_Parking-Sign.pt', '/net/acadia14a/data/user/Relabeling/dinov2_data/parking_signs/facebook_dinov2-giant_parking_signs.pt'],
#  'No parking': ['/net/acadia14a/data/user/Relabeling/dinov2_data/traffic_signs/train/cropped_images/noparking/facebook_dinov2-giant_noparking.pt']}

# turn_signs = ['mand_left', 'mand_left_right', 'mand_pass_left', 'mand_pass_left_right', 'mand_pass_right', 'mand_right', 'mand_straigh_left', 'mand_straight', 'mand_straight_right']
# turn_signs = [os.path.join('/net/acadia14a/data/user/Relabeling/dinov2_data/Traffic Signs Detection Europe_v14-traffic-signs-detection-4_5k-images_coco/train/cropped_images',i, 'facebook_dinov2-giant_%s.pt'%i) for i in turn_signs]
# class_files['Turn signs'] = turn_signs
class_features = {}

for class_name, pt_files in class_files.items():
    # Load all features for the current class and store them in a list
    loaded_features = [torch.load(pt_file) for pt_file in pt_files]

    # Concatenate all loaded features along the batch dimension
    if loaded_features:  # Check if there are any loaded features
        stacked_features = torch.cat(loaded_features, dim=0)
        class_features[class_name] = stacked_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load DINOv2 model and image processor
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
        return None, None
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
        return None, None
    if top_two and top_classes:
        for j, (class_name, similarity) in enumerate(top_classes):
            usdot_class = class_name
            label = f'{usdot_class}: {similarity:.2f}'
    else:
        usdot_class = best_class
        label = f'{usdot_class}: {score:.2f}'
    return usdot_class, score

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
        dino_response, score = return_label(cropped_image, class_features) 
        gt = ground_truth.lower()
        print("Prediction1:", dino_response, " gt:", gt)
        img_count+=1
        if dino_response:
            if dino_response.lower() == gt:
                correct_vlm1+=1
            else:
                print(score)
                cropped_image.save('wrong/%s.png'%str(img_count))
        else:
            if gt == 'others':
                correct_vlm1+=1
        total+=1

    save_path = 'dino_v2'
    with open('%s_signs.txt'%save_path, 'w') as f:
        f.write(str(correct_vlm1/total))
        f.write('\n')

if __name__ == "__main__":
    main()

