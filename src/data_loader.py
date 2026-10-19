import json
import os
from pathlib import Path
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

from src.config import (
    CLASS_LABELS_PATH,
    DATA_DIR,
    IMG_SIZE,
    RANDOM_STATE,
    VALID_EXT,
)


def get_class_names(data_dir=DATA_DIR):
    return sorted([
        folder.name
        for folder in Path(data_dir).iterdir()
        if folder.is_dir()
    ])


def load_raw_data(data_dir=DATA_DIR, img_size=IMG_SIZE, valid_ext=VALID_EXT):
    data_dir = Path(data_dir)
    class_names = get_class_names(data_dir)
    X = []
    labels = []

    for class_name in class_names:
        class_dir = data_dir / class_name

        for image_path in class_dir.iterdir():
            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in valid_ext:
                continue

            try:
                image = cv2.imread(str(image_path))
                if image is None:
                    print(f"Could not load: {image_path}")
                    continue

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, img_size)

                X.append(image)
                labels.append(class_name)

            except (cv2.error, OSError, ValueError) as error:
                print(f"Error with {image_path}: {error}")

        print(f"Finished class: {class_name}")

    print("\nTotal images:", len(X))
    print("Total labels:", len(labels))
    if X:
        print("Image shape:", X[0].shape)
        print("First label:", labels[0])

    X = np.array(X, dtype=np.uint8)
    y = np.array(labels)
    return X, y, class_names


def preprocess_and_split(X, labels, class_names=None, random_state=RANDOM_STATE):
    X = X.astype(np.float32) / 255.0
    print("Minimum pixel value:", X.min())
    print("Maximum pixel value:", X.max())

    label_encoder = LabelEncoder()
    y_integers = label_encoder.fit_transform(labels)

    class_mapping = {int(i): str(c) for i, c in enumerate(label_encoder.classes_)}
    os.makedirs(CLASS_LABELS_PATH.parent, exist_ok=True)
    with open(CLASS_LABELS_PATH, "w") as f:
        json.dump(class_mapping, f, indent=4)
    print("Saved class mapping:", class_mapping)

    num_classes = len(class_mapping)

    X_train_val, X_test, y_train_val_idx, y_test_idx = train_test_split(
        X, y_integers, test_size=0.20, random_state=random_state, stratify=y_integers, shuffle=True
    )

    X_train, X_val, y_train_idx, y_val_idx = train_test_split(
        X_train_val, y_train_val_idx, test_size=0.125, random_state=random_state, stratify=y_train_val_idx, shuffle=True
    )

    y_train = to_categorical(y_train_idx, num_classes=num_classes)
    y_val = to_categorical(y_val_idx, num_classes=num_classes)
    y_test = to_categorical(y_test_idx, num_classes=num_classes)

    print(f"Train set: {X_train.shape} | Labels: {y_train.shape}")
    print(f"Val set:   {X_val.shape} | Labels: {y_val.shape}")
    print(f"Test set:  {X_test.shape} | Labels: {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test, label_encoder


if __name__ == "__main__":
    X, y, class_names = load_raw_data()
    X_train, y_train, X_val, y_val, X_test, y_test, _ = preprocess_and_split(X, y, class_names)
