from flask import Flask, request, jsonify, render_template
import torch
import numpy as np
import os
import json

from model import EEG_ASD_Model
from preprocess import load_eeg, create_windows

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = "cpu"

# Load model
model = EEG_ASD_Model()
model.load_state_dict(torch.load("artifacts/model.pth", map_location=device))
model.eval()

with open("artifacts/label_map.json") as f:
    LABEL_MAP = json.load(f)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    files = request.files.getlist("files")
    if len(files) < 1:
        return render_template("index.html", error="No files uploaded")

    saved_files = {}
    for f in files:
        path = os.path.join(UPLOAD_FOLDER, f.filename)
        f.save(path)
        saved_files[f.filename.split(".")[-1]] = path

    if "set" not in saved_files:
        return render_template("index.html", error="Missing .set file")

    # IMPORTANT: .fdt must be in SAME folder with SAME base name
    eeg = load_eeg(saved_files["set"])
    windows = create_windows(eeg)

    if len(windows) == 0:
        return render_template("index.html", error="EEG signal too short")

    X = torch.tensor(windows)

    with torch.no_grad():
        preds = model(X).argmax(1).numpy()

    final_pred = int(np.bincount(preds).argmax())
    confidence = (preds == final_pred).mean()

    return render_template(
        "index.html",
        prediction=LABEL_MAP[str(final_pred)],
        confidence=round(confidence * 100, 2),
        windows=len(windows)
    )



if __name__ == "__main__":
    app.run(debug=True)
