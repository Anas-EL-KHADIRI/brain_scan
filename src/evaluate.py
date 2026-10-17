import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf

from src.config import CLASS_NAMES, MODEL_PATH, REPORTS_DIR
from src.data_loader import load_raw_data, preprocess_and_split
from src.visualization import plot_confusion_matrix, plot_predictions


def evaluate_model(model_path=MODEL_PATH, X_test=None, y_test=None, class_names=CLASS_NAMES):
    # If test data not passed, load it
    if X_test is None or y_test is None:
        print("Loading test data...")
        X, y, class_names = load_raw_data()
        _, _, _, _, X_test, y_test, _ = preprocess_and_split(X, y, class_names)

    print(f"\nLoading trained model from: {model_path}")
    best_model = tf.keras.models.load_model(str(model_path))

    # Quantitative test evaluation (Notebook Cell 20)
    test_loss, test_accuracy = best_model.evaluate(X_test, y_test, verbose=1)
    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.2%}")

    # Predict probabilities for every test image (Notebook Cell 21)
    y_prob = best_model.predict(X_test, verbose=1)

    # Convert one-hot labels and probabilities to class indices
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_prob, axis=1)

    # Confusion matrix (Notebook Cell 21)
    cm_path = REPORTS_DIR / "confusion_matrix.png"
    cm = plot_confusion_matrix(y_true, y_pred, class_names=class_names, save_path=cm_path)
    print(f"\nSaved confusion matrix to: {cm_path}")

    # Classification report (Notebook Cell 21)
    print("\nClassification Report:")
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    print(report)

    # Correct and incorrect prediction galleries (Notebook Cell 22)
    correct = np.where(y_true == y_pred)[0]
    incorrect = np.where(y_true != y_pred)[0]

    print("Correct predictions count:", len(correct))
    print("Incorrect predictions count:", len(incorrect))

    plot_predictions(
        X_test, y_true, y_pred, class_names=class_names,
        indices=correct, title="Correct Predictions",
        save_path=REPORTS_DIR / "correct_predictions.png"
    )

    if len(incorrect) > 0:
        plot_predictions(
            X_test, y_true, y_pred, class_names=class_names,
            indices=incorrect, title="Incorrect Predictions (Error Analysis)",
            save_path=REPORTS_DIR / "incorrect_predictions.png"
        )

    return test_loss, test_accuracy, report, cm


if __name__ == "__main__":
    evaluate_model()
