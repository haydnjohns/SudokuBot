import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load and preprocess the data
# -------------------------------

# Load dataset from .npz file
data = np.load('sudoku_digits.npz')
x_train = data['x_train']
y_train = data['y_train']

# Normalize and reshape for CNN
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# One-hot encode labels
y_train_cat = to_categorical(y_train-1, num_classes=9)

# Train-validation split
x_train_split, x_val, y_train_split, y_val = train_test_split(
    x_train, y_train_cat, test_size=0.1, random_state=42
)

# -------------------------------
# 2. Define the CNN model
# -------------------------------

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(9, activation='softmax')  # 9 classes (digits 1–9)
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# 3. Train the model
# -------------------------------

history = model.fit(
    x_train_split, y_train_split,
    validation_data=(x_val, y_val),
    epochs=15,
    batch_size=64,
    verbose=1
)

# -------------------------------
# 4. Evaluate the model
# -------------------------------

loss, acc = model.evaluate(x_val, y_val, verbose=0)
print(f"Validation accuracy: {acc:.4f}")

# -------------------------------
# 5. Save the model
# -------------------------------

model.save("digit_cnn.keras")
print("Model saved as digit_cnn.keras")

# -------------------------------
# 6. (Optional) Plot training history
# -------------------------------

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label="Train Acc")
plt.plot(history.history['val_accuracy'], label="Val Acc")
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.title("Loss")
plt.legend()

plt.tight_layout()
plt.show()