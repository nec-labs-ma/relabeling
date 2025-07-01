import os
import json
import cv2

def save_cropped_images(annotations_path, images_dir, output_dir, target_labels):
    """
    Extracts and saves cropped images based on given labels from COCO-format Object365 data.

    Args:
        annotations_path (str): Path to the COCO JSON annotation file.
        images_dir (str): Directory containing original images.
        output_dir (str): Directory where cropped images will be saved.
        target_labels (list): List of labels to extract and save.

    Returns:
        None
    """
    # Load COCO JSON annotations
    with open(annotations_path, "r") as f:
        coco_data = json.load(f)

    # Create a mapping from category ID to category name
    category_mapping = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

    # Reverse mapping (label -> category_id)
    label_to_category_id = {v: k for k, v in category_mapping.items()}

    # Filter annotations based on target labels
    target_category_ids = {label_to_category_id[label] for label in target_labels if label in label_to_category_id}

    # Create a mapping from image ID to file name
    image_id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}

    # Create output directories for each label
    for label in target_labels:
        os.makedirs(os.path.join(output_dir, label), exist_ok=True)

    # Iterate over annotations and crop images
    for ann in coco_data["annotations"]:
        category_id = ann["category_id"]
        if category_id in target_category_ids:
            image_id = ann["image_id"]
            bbox = ann["bbox"]  # Format: [x, y, width, height]

            # Load the corresponding image
            image_filename = image_id_to_filename.get(image_id)
            if not image_filename:
                continue

            image_path = os.path.join(images_dir, image_filename)
            if not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found.")
                continue

            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Failed to load image {image_path}.")
                continue

            # Crop the object
            x, y, w, h = map(int, bbox)
            cropped_image = image[y:y+h, x:x+w]

            # Skip empty crops
            if cropped_image.size == 0:
                continue

            # Save the cropped image in the corresponding label folder
            label_name = category_mapping[category_id]
            save_path = os.path.join(output_dir, label_name, f"{image_id}_{ann['id']}.jpg")
            cv2.imwrite(save_path, cropped_image)

            print(f"Saved: {save_path}")

    print("Cropping and saving completed!")

annotations_path = '/net/acadia12a/data/user/objects365v1/zsy_objv1_train_xdino.json'
images_dir = '/net/acadia12a/data/user/objects365v1/train'
output_dir = '/net/acadia14a/data/user/Relabeling/dinov2_data/new_vehicles'
target_labels = [
    'car', 'suv', 'van', 'bus', 'motorcycle', 'bicycle', 'pickup truck', 'truck',
    'machinery vehicle', 'sports car', 'fire truck', 'heavy truck', 'ambulance',
    'tricycle', 'rickshaw', 'trolley', 'carriage', 'formula 1 car', 'hoverboard',
    'wheelchair', 'scooter', 'airplane', 'helicopter', 'hot air ballon',
    'boat', 'sailboat', 'ship'
]
save_cropped_images(annotations_path, images_dir, output_dir, target_labels)
