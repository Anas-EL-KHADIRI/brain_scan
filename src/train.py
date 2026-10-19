import time
from src.config import BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_PATH, REPORTS_DIR
from src.data_loader import load_raw_data, preprocess_and_split
from src.model import build_cnn_model, compile_model, get_callbacks
from src.visualization import plot_learning_curves


def train_pipeline(epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, model_path=MODEL_PATH):
    print("1. Loading raw dataset...")
    X, y, class_names = load_raw_data()

    print("2. Preprocessing and splitting data...")
    X_train, y_train, X_val, y_val, X_test, y_test, _ = preprocess_and_split(X, y, class_names)

    print("3. Building CNN architecture...")
    model = build_cnn_model(input_shape=(224, 224, 3), num_classes=len(class_names))
    model.summary()

    print("4. Compiling model with Adam & Callbacks...")
    compile_model(model, learning_rate=learning_rate)
    callbacks = get_callbacks(model_path=model_path)

    print(f"5. Starting CNN training (batch_size={batch_size}, epochs={epochs})...")
    start_time = time.time()

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    seconds_total = time.time() - start_time
    print("Training time:", int(seconds_total // 60), "min", int(seconds_total % 60), "sec")

    learning_curves_path = REPORTS_DIR / "learning_curves.png"
    plot_learning_curves(history, save_path=learning_curves_path)
    print(f"Saved learning curves to: {learning_curves_path}")

    return model, history, (X_test, y_test)


if __name__ == "__main__":
    train_pipeline()
