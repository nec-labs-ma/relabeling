import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import pickle
from PIL import Image
from collections import Counter
import numpy as np

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration
batch_size = 32
num_epochs = 50
model_save_path = "resnet101_bdd_vehicles_classifier.pth"
class_to_idx_path = "bdd_vehicles_class_to_idx.pkl"

# Define class_files manually or load from config
class_files = {
    # -- class 1
    "car": ["/net/acadia14a/data/user/Relabeling/dinov2_data/new_vehicles/car/facebook_dinov2-giant_car.pt"],
    "bus": ["/net/acadia14a/data/user/Relabeling/dinov2_data/new_vehicles/bus/facebook_dinov2-giant_bus.pt"],
    'truck': ['/net/acadia14a/data/user/Relabeling/dinov2_data/new_vehicles/truck/facebook_dinov2-giant_truck.pt'],
    "motorcycle":['/net/acadia14a/data/user/Relabeling/dinov2_data/new_vehicles/motorcycle/facebook_dinov2-giant_motorcycle.pt'],
    'bicycle': ['/net/acadia14a/data/user/Relabeling/dinov2_data/new_vehicles/bicycle/facebook_dinov2-giant_bicycle.pt'],
    "van":["/net/acadia14a/data/user/Relabeling/dinov2_data/new_vehicles/van/facebook_dinov2-giant_van.pt"],
    "suv":["/net/acadia14a/data/user/Relabeling/dinov2_data/new_vehicles/suv/facebook_dinov2-giant_suv.pt"],
}

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Custom Dataset
class VehicleCropDataset(Dataset):
    def __init__(self, class_files, transform=None):
        self.samples = []
        self.class_to_idx = {}
        self.transform = transform

        for idx, (label, pt_paths) in enumerate(class_files.items()):
            self.class_to_idx[label] = idx
            for pt_path in pt_paths:
                img_dir = Path(pt_path).parent
                jpgs = list(img_dir.glob("*.jpg"))
                if not jpgs:
                    print(f"⚠️ No .jpg images found in {img_dir}. Skipping.")
                    continue
                for jpg in jpgs:
                    self.samples.append((jpg, idx))

        print(f"✅ Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Load dataset and dataloader
dataset = VehicleCropDataset(class_files, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Compute class weights
labels = [label for _, label in dataset.samples]
label_counts = Counter(labels)
total = sum(label_counts.values())
weights = [total / label_counts[i] for i in range(len(label_counts))]
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

# Define model
model = models.resnet101(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(dataset.class_to_idx))
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
print("Training...")
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")

# Save model and class map
torch.save({
    'model_state_dict': model.state_dict(),
    'class_to_idx': dataset.class_to_idx
}, model_save_path)

with open(class_to_idx_path, 'wb') as f:
    pickle.dump(dataset.class_to_idx, f)

print(f"✅ Model saved to {model_save_path}")
print(f"✅ Class to index mapping saved to {class_to_idx_path}")
