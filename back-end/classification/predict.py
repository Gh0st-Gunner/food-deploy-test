import torch
import numpy as np
from torchvision import transforms
from PIL import Image


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def predict(image, model, class_names, device, top_k=5):
    transform = get_transform()
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, top_k)
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]

    results = []
    for i in range(top_k):
        results.append({
            "class": class_names[top_indices[i]],
            "probability": float(top_probs[i]),
            "rank": i + 1,
        })
    return results


def predict_onnx(image, session, input_name, class_names, top_k=5):
    transform = get_transform()
    x = transform(image).unsqueeze(0).numpy().astype(np.float32)
    raw = session.run(None, {input_name: x})[0]
    e = np.exp(raw - np.max(raw, axis=1, keepdims=True))
    probs = (e / e.sum(axis=1, keepdims=True))[0]
    top_indices = np.argsort(probs)[::-1][:top_k]
    return [
        {"class": class_names[idx], "probability": float(probs[idx]), "rank": rank}
        for rank, idx in enumerate(top_indices, 1)
    ]