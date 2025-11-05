import io
from pathlib import Path
import streamlit as st
import torch
from PIL import Image
import numpy as np

import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

MODEL_PATH = Path(__file__).parent / "best_resnet50_multilabel.pth"

st.title("ResNet50 Multi-label Classifier")

@st.cache_resource
def load_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(path, map_location="cpu")
    label_cols = None
    # If a full model was saved
    if isinstance(data, nn.Module):
        model = data
    else:
        # If checkpoint dict (state_dict) or wrapped checkpoint
        if isinstance(data, dict):
            # common keys used when saving checkpoints
            if "state_dict" in data:
                state_dict = data["state_dict"]
            elif "model_state_dict" in data:
                # some scripts save under 'model_state_dict'
                state_dict = data["model_state_dict"]
            elif "model" in data and isinstance(data["model"], dict):
                state_dict = data["model"]
            else:
                # assume the dict itself is the state_dict
                state_dict = data
            # capture any label columns saved in the checkpoint
            label_cols = data.get("label_cols") if "label_cols" in data else data.get("labels") if "labels" in data else None
        else:
            state_dict = data
        # infer number of classes from fc weight if available
        out_features = None
        if "fc.weight" in state_dict:
            out_features = state_dict["fc.weight"].shape[0]
        elif "module.fc.weight" in state_dict:
            out_features = state_dict["module.fc.weight"].shape[0]
        # build resnet50 and adapt final layer if needed
        model = models.resnet50(pretrained=False)
        if out_features is not None:
            model.fc = nn.Linear(model.fc.in_features, out_features)
        # try loading state dict (handle possible "module." prefixes)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            # strip "module." if present
            new_state = {}
            for k, v in state_dict.items():
                nk = k.replace("module.", "") if k.startswith("module.") else k
                new_state[nk] = v
            model.load_state_dict(new_state)
    model.eval()
    model.to(device)
    return model, device, label_cols

def preprocess_image(image: Image.Image):
    transform = T.Compose([
        T.Resize((200, 200)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    if image.mode != "RGB":
        image = image.convert("RGB")
    return transform(image)

# UI: model load status
if not MODEL_PATH.exists():
    st.error(f"Model file not found at {MODEL_PATH}")
    st.stop()

with st.spinner("Loading model..."):
    model, device, label_cols = load_model(str(MODEL_PATH))

st.success("Model loaded")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
labels_value = "\n".join(label_cols) if label_cols else ""
labels_text = st.text_area("Optional labels (one per line). If empty, classes will be numbered.", height=120, value=labels_value)
threshold = st.slider("Probability threshold (for showing predicted classes)", 0.0, 1.0, 0.5, 0.01)
top_k = st.number_input("Show top K predictions", min_value=1, max_value=100, value=10, step=1)

if uploaded is not None:
    image = Image.open(io.BytesIO(uploaded.read()))
    st.image(image, caption="Input image", use_column_width=True)
    input_tensor = preprocess_image(image).unsqueeze(0).to(device)

    with st.spinner("Running inference..."):
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy().squeeze()

    # labels handling
    labels = []
    if labels_text.strip():
        # accept newline or comma separated
        if "\n" in labels_text.strip():
            labels = [l.strip() for l in labels_text.strip().splitlines() if l.strip()]
        else:
            labels = [l.strip() for l in labels_text.strip().split(",") if l.strip()]
    # fallback to numeric labels
    if not labels:
        labels = [f"class_{i}" for i in range(probs.shape[0])]

    if probs.shape[0] != len(labels):
        st.warning("Number of provided labels does not match model outputs. Falling back to numeric labels.")
        labels = [f"class_{i}" for i in range(probs.shape[0])]

    # assemble results
    idxs = np.argsort(probs)[::-1][:int(top_k)]
    results = []
    for i in idxs:
        results.append({"label": labels[i], "probability": float(probs[i])})

    st.subheader("Top predictions")
    for r in results:
        status = "✔" if r["probability"] >= threshold else ""
        st.write(f"{r['label']}: {r['probability']:.4f} {status}")

    # show all above threshold
    above = [(labels[i], float(probs[i])) for i in range(len(probs)) if probs[i] >= threshold]
    if above:
        st.subheader(f"Classes above threshold ({threshold})")
        for lab, p in sorted(above, key=lambda x: x[1], reverse=True):
            st.write(f"{lab}: {p:.4f}")
    else:
        st.info("No classes above the threshold.")