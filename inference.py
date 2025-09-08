import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import numpy as np
import time
import os

class CountryClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CountryClassifier, self).__init__()
        self.backbone = timm.create_model('convnext_tiny', pretrained=False, num_classes=0)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

def load_model(checkpoint_path, num_classes, device):
    model = CountryClassifier(num_classes=num_classes).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    idx_to_class = checkpoint['idx_to_class']

    return model, idx_to_class

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def predict_single_image(image_path, checkpoint_path="models/country_detection_model_checkpoint.pth", device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_classes = checkpoint['num_classes']
    model, idx_to_class = load_model(checkpoint_path, num_classes, device)
    print(f"Model loaded on {device}, num_classes == {num_classes}")

    image = Image.open(image_path).convert('RGB')
    print(f"Image loaded: {image.size}")

    transform = get_transform()
    image_tensor = transform(image).unsqueeze(0).to(device)

    start_time = time.time()
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    inference_time = (time.time() - start_time) * 1000

    pred_country = idx_to_class[pred_idx.item()]
    confidence = confidence.item()

    result = {
        'country': pred_country,
        'confidence': confidence,
        'inference_time_ms': inference_time
    }

    print("\n" + "="*50)
    print("Inference results:")
    print("="*50)
    print(f"Country: {result['country']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Time: {result['inference_time_ms']:.1f} ms")
    print("="*50)

    return result

result = predict_single_image("demo\us.jpeg", checkpoint_path="models\country_detection_model_checkpoint.pth")