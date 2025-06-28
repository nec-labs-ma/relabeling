import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

sns.set(style="whitegrid")

def analyze_type_of_signs(json_path, output_dir="type_sign_analysis"):
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    images = data["images"]
    annotations = data["annotations"]

    # Stats
    type_count = defaultdict(int)
    type_area = defaultdict(list)
    type_to_images = defaultdict(set)
    type_occluded = defaultdict(int)

    for ann in tqdm(annotations, desc="Processing annotations"):
        attr = ann.get("attributes", {})
        type_of_sign = attr.get("Type of Signs", "Unknown").strip()
        if type_of_sign == "- Others":
            continue  # ❌ Exclude this category
        type_of_sign = type_of_sign.replace('- ', '')
        is_occluded = attr.get("occluded", False)

        type_count[type_of_sign] += 1
        type_area[type_of_sign].append(ann["area"])
        type_to_images[type_of_sign].add(ann["image_id"])
        if is_occluded:
            type_occluded[type_of_sign] += 1

    # Bar chart: annotations per type
    plt.figure(figsize=(10, 5))
    types = list(type_count.keys())
    counts = [type_count[t] for t in types]
    sns.barplot(x=types, y=counts)
    plt.title("Annotations per 'Type of Signs'")
    plt.ylabel("Count")
    plt.xlabel("Type of Sign")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "annotations_per_type.png"))
    plt.close()

    # Bar chart: images per type
    plt.figure(figsize=(10, 5))
    img_counts = [len(type_to_images[t]) for t in types]
    sns.barplot(x=types, y=img_counts, palette="Set2")
    plt.title("Unique Images per 'Type of Signs'")
    plt.ylabel("Image Count")
    plt.xlabel("Type of Sign")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "images_per_type.png"))
    plt.close()

    # Bar chart: average area per type
    plt.figure(figsize=(10, 5))
    avg_areas = [np.mean(type_area[t]) for t in types]
    sns.barplot(x=types, y=avg_areas, palette="coolwarm")
    plt.title("Average Bbox Area per 'Type of Signs'")
    plt.ylabel("Area (px²)")
    plt.xlabel("Type of Sign")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "avg_bbox_area_per_type.png"))
    plt.close()

    # Bar chart: occlusion rate
    plt.figure(figsize=(10, 5))
    occlusion_rates = [type_occluded[t] / type_count[t] for t in types]
    sns.barplot(x=types, y=occlusion_rates, palette="muted")
    plt.title("Occlusion Rate per 'Type of Signs'")
    plt.ylabel("Fraction occluded")
    plt.xlabel("Type of Sign")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "occlusion_rate_per_type.png"))
    plt.close()

    print(f"✅ Saved analysis to: {output_dir}/")

# Example usage:
# analyze_type_of_signs("path/to/annotations.json")


# Example usage:
analyze_type_of_signs("/net/acadia14a/data/sparsh/Relabeling/instances_default.json")

