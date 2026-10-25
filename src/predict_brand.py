
from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "car_brand_classifier.pth"
IMAGE_DIR = PROJECT_DIR / "data" / "sample_cars"

device = torch.device("cpu")
IMAGE_SIZE = 224

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

class_names = checkpoint["class_names"]

model = models.mobilenet_v3_small(weights=None)
in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, len(class_names))
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

if not IMAGE_DIR.exists():
    raise FileNotFoundError(f"Image folder not found: {IMAGE_DIR}")

image_files = [
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
]

if not image_files:
    raise FileNotFoundError(f"No images found in {IMAGE_DIR}")

image_path = image_files[0]
image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(image_tensor)
    probabilities = torch.softmax(outputs, dim=1)[0]
    confidence, predicted_index = probabilities.max(dim=0)

print("Image:", image_path.name)
print("Predicted brand:", class_names[predicted_index.item()])
print(f"Model confidence: {confidence.item():.1%}")
print("\nNote: This is a baseline prediction, not yet a complete car detector.")
