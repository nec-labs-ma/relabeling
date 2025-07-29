# 🛑 Relabeled Mapillary Traffic Sign Validation Dataset (2000 Images)

Welcome to the official repository for our relabeled **Mapillary validation dataset**, focused on **traffic sign understanding**. In this project, we release **2,000 curated images** from the Mapillary dataset with **manually re-annotated traffic sign labels**, structured into a new, cleaner label taxonomy.

---

## 📌 Project Overview (Published in ICCV Workshop)

This project aims to **standardize and improve** traffic sign annotations by relabeling existing Mapillary data using **modern Visual-Language Models (VLMs)** like:

- [InternVL](https://github.com/OpenGVLab/InternVL)
- [Gemma](https://ai.google.dev/gemma)
- and other open-source foundation models

These models were leveraged to assist human annotators in refining the dataset, improving label clarity and coverage while reducing noise.

---

## 🗂️ Dataset Description

We relabeled **2,000 validation images** from Mapillary with a focus on **traffic sign recognition**, especially relevant for **autonomous driving** and **urban scene understanding**.

### ✅ New Label Taxonomy

Each traffic sign in the dataset is annotated into one of the following **12 categories**:

- `stop signs`
- `speed limit signs`
- `yield signs`
- `do not enter`
- `crosswalk signs`
- `parking signs`
- `no parking`
- `roundabout signs`
- `turn signs`
- `cycle lane signs`
- `others`

---

## 🔍 Motivation

While Mapillary provides a large and diverse dataset, the traffic sign annotations are often **coarse or inconsistent**. Our goal is to:

- Create a **cleaner, task-specific benchmark** for traffic sign detection and classification
- Enable **VLM-based semi-automated relabeling** pipelines
- Serve as a reference for **fine-tuning or evaluating vision-language models** on structured scene understanding

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone (https://github.com/nec-labs-ma/relabeling.git)](https://github.com/nec-labs-ma/relabeling.git)
cd relabeling
```

### 2. Install requirements

```bash
pip install -r requirements.txt
pip install awscli==1.25.0
```

### 3. Download the dataset
- Download images here: `https://mapillary-signs.s3.us-west-2.amazonaws.com/images.zip` or `aws s3 cp s3://mapillary-signs/images.zip .`
- Download annotations for signs: `https://mapillary-signs.s3.us-west-2.amazonaws.com/instances_default.json` or `aws s3 cp s3://mapillary-signs/instances_default.json .`

---

# VLM-Based Dataset Relabeling

This repository supports batch relabeling of vision datasets using multiple variants of **InternVL-3** and **Gemma-3** Visual Language Models (VLMs). It covers four tasks across the Mapillary and BDD100K datasets.

## 📌 Tasks

The relabeling scripts support the following visual tasks:

- **Mapillary Vehicles**
- **Mapillary Pedestrians**
- **Mapillary Traffic Signs**
- **BDD Vehicles**

## 📁 File Structure

```bash
.
├── Relabeling_gemmi3-bdd_vehicles.py
├── Relabeling_gemmi3-mapillary_vehicles.py
├── Relabeling_gemmi3-mapillary_signs.py
├── Relabeling_gemmi3-pedestrains.py
├── Relabeling_internvl-bdd_vehicles.py
├── Relabeling_internvl-mapillary_vehicles.py
├── Relabeling_internvl-mapillary_signs.py
├── Relabeling_internvl-pedestrains.py
└── load_parallel_jobs.sh
```

Each script expects a **single model name** as a command-line argument.

## 🧠 Supported Models

### InternVL-3
- `OpenGVLab/InternVL3-1B`
- `OpenGVLab/InternVL3-2B`
- `OpenGVLab/InternVL3-8B`
- `OpenGVLab/InternVL3-9B`
- `OpenGVLab/InternVL3-14B`

### Gemma-3
- `google/gemma-3-4b-it`
- `google/gemma-3-12b-it`

---

## 🚀 Running the Jobs

Jobs are dispatched using a Slurm batch script `load_parallel_jobs.sh`.

### 🔧 Configuration Steps

1. **Edit `models` array** in `load_parallel_jobs.sh` to include desired model variants.
2. **Replace the Python script name** in the `sbatch` `--wrap=` section according to the task and model family (Gemma or InternVL).
3. **Submit jobs** with:

```bash
bash load_parallel_jobs.sh
```

### 🖥 Example: Relabel with InternVL-3 9B for BDD Vehicles

```bash
models=(
  "OpenGVLab/InternVL3-9B"
)

# In load_parallel_jobs.sh:
--wrap="python Relabeling_internvl-bdd_vehicles.py ${model}"
```

### 🖥 Example: Relabel with Gemma-3 12B for Mapillary Signs

```bash
models=(
  "google/gemma-3-12b-it"
)

# In load_parallel_jobs.sh:
--wrap="python Relabeling_gemmi3-mapillary_signs.py ${model}"
```

---

## 📦 Output

All Slurm logs are saved under the `logs/` directory. Each model variant generates updated labels or predictions for the task-specific dataset in .txt and .json format

```bash
logs/
├── OpenGVLab-InternVL3-9B_bdd.out
├── OpenGVLab-InternVL3-9B_bdd.err
...
```

---

# 🧠 Classifier-Based Relabeling (ResNet50 / ResNet101)

In addition to VLMs, this repository includes support for relabeling using ResNet-based classifiers for all four tasks.

## 🔧 Tasks

- **Mapillary Vehicles**
- **Mapillary Pedestrians**
- **Mapillary Traffic Signs**
- **BDD Vehicles**

## 🏋️ Training

Trainer scripts are available for each task:

```bash
bdd_vehicles_classifier.py
humans_classifier.py
mapillary_vehicles_classifier.py
signs_classifier.py
```

Each script trains a ResNet-50 or ResNet-101 model and saves:

- `classifier.pth`: The trained PyTorch model weights
- `label_mapping.pkl`: A mapping of class indices to labels

## 🧪 Inference

After training, use the corresponding inference script:

```bash
bdd_vehicles_classifier_inference.py
human_classifier_inference.py
mapillary_vehicles_classifier_inference.py
signs_classifier_inference.py
```

Make sure to specify the correct path to the `classifier.pth` and `label_mapping.pkl` files.

## 📂 Output

Each inference script generates updated labels or predictions for the task-specific dataset in .txt and .json format.

---

# 🌐 External Data Integration & DINOv2 Feature Extraction

This repository also supports using external datasets (e.g., **Roboflow** and **Object365**) to enhance performance through DINOv2-based feature extraction.

## 📦 External Data Sources

### 🧰 Roboflow

1. Download the relevant dataset ZIP files from Roboflow.
```bash
Traffic Signs: https://universe.roboflow.com/ai-camp-weekend-t3odm/traffic-signs-detection-dpnpl
               https://universe.roboflow.com/radu-oprea-r4xnm/traffic-signs-detection-europe
               https://universe.roboflow.com/kendrickxy/european-road-signs
MotorCycles: https://universe.roboflow.com/cc-kzuq0/helmeteeeeeeeee
Person: https://universe.roboflow.com/mochammad-giri-wiwaha-ngulandoro/person-vthiu
Cyclists: https://universe.roboflow.com/bicycle-detection/bike-detect-ct
Pedestrians: https://universe.roboflow.com/erickson49366-gmail-com/cyclist-detector-training-data-v3
```
2. List the paths to these ZIPs in `extract_boxes.py`.
3. Run the script to extract and crop object instances:
   ```bash
   python extract_boxes.py
   ```

### 🗂 Object365

1. Use `fetch_object_365.py` to extract and crop objects:
   ```bash
   python fetch_object_365.py
   ```

All cropped object images will be saved in task-specific directories.

---

## 📌 DINOv2 Feature Extraction

1. Use `extract_features.py` to compute DINOv2 features for cropped objects (e.g., speed signs, yield signs).
2. Features will be saved as:
   ```bash
   facebook_dinov2-giant_<object>.pt
   ```

---

## 🧠 Using DINOv2 Features

Use the extracted `.pt` feature files as input to the following scripts via the `class_features` parameter:

```bash
dinov2_bdd_vehicles.py
dinov2_humans.py
dinov2_mapillary_signs.py
dinov2_mapillary_vehicles.py
```

Run these scripts after passing the appropriate paths to the precomputed class features.

---


## 📜 License

This dataset is released for **research and academic use only**. Please check `LICENSE.txt` for details.

---

## 🙌 Acknowledgements

- Mapillary Vistas Dataset
- InternVL by OpenGVLab
- Gemma by Google DeepMind

---
