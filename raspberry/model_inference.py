import json
from pathlib import Path

from PIL import Image
import torch
from torchvision import transforms


def load_classifier(model_path="deploy/models/efficientnet_b0_rpi.pt", meta_path="deploy/models/efficientnet_b0_meta.json"):
    model = torch.jit.load(str(model_path), map_location="cpu")
    model.eval()
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    classes = meta["classes"]
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, classes, preprocess


def predict_image(image_path, model, classes, preprocess):
    img = Image.open(image_path).convert("RGB")
    with torch.no_grad():
        probs = torch.softmax(model(preprocess(img).unsqueeze(0)), dim=1)[0]
    confidence, index = torch.max(probs, dim=0)
    return {"class": classes[int(index)], "confidence": float(confidence)}
