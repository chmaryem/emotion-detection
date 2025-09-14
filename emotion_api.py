from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import os
from timm import create_model
from timm.layers import DropPath

app = FastAPI()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model + TTA
class FERModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.backbone = create_model("convnext_tiny", pretrained=False, num_classes=0, drop_rate=0.3)
        self.drop_path = DropPath(drop_prob=0.1)
        self.head = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.drop_path(x)
        return self.head(x)

@torch.no_grad()
def predict_with_tta(model, image, n_aug=5):
    model.eval()
    preds = []
    preds.append(model(image.unsqueeze(0)).softmax(dim=1))

    for _ in range(n_aug - 1):
        aug_img = image.clone()
        if np.random.rand() > 0.5:
            aug_img = torch.flip(aug_img, dims=[2])
        output = model(aug_img.unsqueeze(0))
        preds.append(output.softmax(dim=1))

    return torch.mean(torch.stack(preds), dim=0)

# Load model
model = FERModel()
checkpoint = torch.load("best_fer_model.pth", map_location=device)
model.load_state_dict(checkpoint["ema"])
model.to(device)
model.eval()

# Transform
val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


class ImagePath(BaseModel):
    image_url: str

@app.post("/analyze-image-url")
def analyze_image(payload: ImagePath) -> Dict[str, str]:
    path = payload.image_url

    if not os.path.exists(path):
        return {"error": f"File not found: {path}"}

    try:
        image = Image.open(path).convert("RGB")
        tensor = val_transform(image).to(device)

        pred = predict_with_tta(model, tensor)
        label = emotion_classes[pred.argmax().item()]
        confidence = pred.max().item()

        return {
            "emotion": label,
            "confidence": f"{confidence:.4f}"
        }

    except Exception as e:
        return {"error": str(e)}
