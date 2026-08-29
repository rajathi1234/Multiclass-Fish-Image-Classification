import os
import numpy as np
import tensorflow as tf

from keras import layers, models
from keras.utils import load_img, img_to_array


# -----------------------------
# 1. Paths
# -----------------------------

train_path = "dataset/fish_dataset/train"
valid_path = "dataset/fish_dataset/valid"


# -----------------------------
# 2. Class names
# -----------------------------

classes = ["Catla", "CommonCarp", "Mori", "Rohu", "SilverCarp"]

class_to_index = {
    name: index for index, name in enumerate(classes)
}


# -----------------------------
# 3. Load images
# -----------------------------

def load_data(folder):

    images = []
    labels = []

    for filename in os.listdir(folder):

        if filename.lower().endswith(".jpg"):

            class_name = filename.split("_")[0]

            if class_name in class_to_index:

                image_path = os.path.join(folder, filename)

                image = load_img(
                    image_path,
                    target_size=(128, 128)
                )

                image = img_to_array(image)

                images.append(image)
                labels.append(class_to_index[class_name])

    return np.array(images), np.array(labels)


print("Loading training data...")
X_train, y_train = load_data(train_path)

print("Loading validation data...")
X_valid, y_valid = load_data(valid_path)

print("Training images:", len(X_train))
print("Validation images:", len(X_valid))


# -----------------------------
# 4. Normalize images
# -----------------------------

X_train = X_train / 255.0
X_valid = X_valid / 255.0


# -----------------------------
# 5. Data Augmentation
# -----------------------------

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])


# -----------------------------
# 6. CNN Model
# -----------------------------

model = models.Sequential([

    data_augmentation,

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(128, 128, 3)
    ),

    layers.MaxPooling2D(2, 2),

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(2, 2),

    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(2, 2),

    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.5),

    layers.Dense(
        5,
        activation="softmax"
    )
])


# -----------------------------
# 7. Compile
# -----------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# -----------------------------
# 8. Show model
# -----------------------------

model.summary()


# -----------------------------
# 9. Train model
# -----------------------------

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_valid, y_valid),
    epochs=15,
    batch_size=32
)


# -----------------------------
# 10. Save model
# -----------------------------

os.makedirs("models", exist_ok=True)

model.save("models/cnn_fish_classifier.keras")

print("CNN model saved successfully!")