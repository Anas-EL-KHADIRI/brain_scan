import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf

from src.config import CLASS_LABELS_PATH, IMG_SIZE, MODEL_PATH

# Cache the loaded model and labels to avoid reloading on each call in apps/APIs
_CACHED_MODEL = None
_CACHED_LABELS = None


def load_inference_artifacts(model_path=MODEL_PATH, class_labels_path=CLASS_LABELS_PATH):
    # Load and cache the trained Keras model and JSON label mapping
    global _CACHED_MODEL, _CACHED_LABELS

    if _CACHED_LABELS is None:
        with open(class_labels_path, "r", encoding="utf-8") as f:
            _CACHED_LABELS = json.load(f)

    if _CACHED_MODEL is None:
        _CACHED_MODEL = tf.keras.models.load_model(str(model_path))

    return _CACHED_MODEL, _CACHED_LABELS


def preprocess_single_image(image_input, img_size=IMG_SIZE):
    # Preprocess single image identically to training (Notebook Cells 5 & 9)
    if isinstance(image_input, (str, Path)):
        image = cv2.imread(str(image_input))
        if image is None:
            raise ValueError(f"Could not read image from path: {image_input}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, np.ndarray):
        image = image_input.copy()
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:
        raise TypeError("image_input must be a file path (str/Path) or a numpy ndarray")

    # Resize to (224, 224)
    image_resized = cv2.resize(image, img_size)

    # Normalize pixels [0, 1]
    image_normalized = image_resized.astype(np.float32) / 255.0

    # Expand batch dimension: (1, 224, 224, 3)
    image_batch = np.expand_dims(image_normalized, axis=0)
    return image_batch, image


def predict_image(image_input, model_path=MODEL_PATH, class_labels_path=CLASS_LABELS_PATH):
    # Predict tumor class and confidence score for a single MRI scan
    model, class_labels = load_inference_artifacts(model_path, class_labels_path)
    image_batch, raw_image = preprocess_single_image(image_input)

    # Predict
    probabilities_raw = model.predict(image_batch, verbose=0)[0]
    predicted_idx = int(np.argmax(probabilities_raw))

    # Format output
    probabilities = {
        class_labels[str(i)]: float(prob)
        for i, prob in enumerate(probabilities_raw)
    }
    predicted_class = class_labels[str(predicted_idx)]
    confidence = float(probabilities_raw[predicted_idx] * 100.0)

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2),
        "probabilities": {k: round(v, 4) for k, v in probabilities.items()},
        "raw_image": raw_image
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BrainScan AI - Single MRI Scan Prediction")
    parser.add_argument("image_path", type=str, help="Path to the MRI scan image")
    args = parser.parse_args()

    result = predict_image(args.image_path)
    print("\n=== BrainScan AI Diagnostic Result ===")
    print(f"Predicted Diagnosis : {result['predicted_class'].upper()}")
    print(f"Confidence Score    : {result['confidence']}%")
    print(f"Probabilities       : {result['probabilities']}")
