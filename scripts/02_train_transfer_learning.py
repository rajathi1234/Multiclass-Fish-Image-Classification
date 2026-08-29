import os
import json
import numpy as np
import tensorflow as tf

from keras import layers, models
from keras.utils import load_img, img_to_array
from keras.applications import (
    VGG16,
    ResNet50,
    MobileNet,
    InceptionV3,
    EfficientNetB0
)


# ==================================================
# 1. Project paths
# ==================================================

TRAIN_PATH = "dataset/fish_dataset/train"
VALID_PATH = "dataset/fish_dataset/valid"

MODEL_PATH = "models"
RESULT_PATH = "results"


# ==================================================
# 2. Classes
# ==================================================

CLASSES = [
    "Catla",
    "CommonCarp",
    "Mori",
    "Rohu",
    "SilverCarp"
]

CLASS_TO_INDEX = {
    name: index for index, name in enumerate(CLASSES)
}


# ==================================================
# 3. Load images
# ==================================================

def load_data(folder):

    images = []
    labels = []

    for filename in os.listdir(folder):

        if filename.lower().endswith(".jpg"):

            class_name = filename.split("_")[0]

            if class_name in CLASS_TO_INDEX:

                image_path = os.path.join(folder, filename)

                image = load_img(
                    image_path,
                    target_size=(128, 128)
                )

                image = img_to_array(image)

                images.append(image)
                labels.append(CLASS_TO_INDEX[class_name])

    return np.array(images), np.array(labels)


print("\nLoading training data...")
X_train, y_train = load_data(TRAIN_PATH)

print("Loading validation data...")
X_valid, y_valid = load_data(VALID_PATH)

print("\nTraining images:", len(X_train))
print("Validation images:", len(X_valid))


# ==================================================
# 4. Normalize images
# ==================================================

X_train = X_train / 255.0
X_valid = X_valid / 255.0


# ==================================================
# 5. Data augmentation
# ==================================================

data_augmentation = tf.keras.Sequential([

    layers.RandomFlip("horizontal"),

    layers.RandomRotation(0.1),

    layers.RandomZoom(0.1)

])


# ==================================================
# 6. Train one model
# ==================================================

def train_model(base_model, model_name):

    print("\n")
    print("=" * 60)
    print("Training", model_name)
    print("=" * 60)

    base_model.trainable = False

    model = models.Sequential([

        data_augmentation,

        base_model,

        layers.GlobalAveragePooling2D(),

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


    # Compile model

    model.compile(

        optimizer="adam",

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]

    )


    # Train model

    history = model.fit(

        X_train,
        y_train,

        validation_data=(
            X_valid,
            y_valid
        ),

        epochs=10,

        batch_size=32

    )


    # Create folders

    os.makedirs(
        MODEL_PATH,
        exist_ok=True
    )

    os.makedirs(
        RESULT_PATH,
        exist_ok=True
    )


    # Save model

    model_file = os.path.join(
        MODEL_PATH,
        f"{model_name}_fish_classifier.keras"
    )

    model.save(model_file)


    # Save training history

    history_file = os.path.join(
        RESULT_PATH,
        f"{model_name}_history.json"
    )

    with open(
        history_file,
        "w"
    ) as file:

        json.dump(
            history.history,
            file
        )


    # Get final results

    final_train_accuracy = history.history["accuracy"][-1]

    final_validation_accuracy = history.history["val_accuracy"][-1]


    print("\n" + "-" * 60)

    print(
        f"{model_name} Final Training Accuracy: "
        f"{final_train_accuracy:.4f}"
    )

    print(
        f"{model_name} Final Validation Accuracy: "
        f"{final_validation_accuracy:.4f}"
    )

    print(
        f"{model_name} model saved successfully!"
    )

    print("-" * 60)


# ==================================================
# 7. VGG16
# ==================================================

vgg16 = VGG16(

    weights="imagenet",

    include_top=False,

    input_shape=(128, 128, 3)

)

train_model(
    vgg16,
    "vgg16"
)


# ==================================================
# 8. ResNet50
# ==================================================

resnet50 = ResNet50(

    weights="imagenet",

    include_top=False,

    input_shape=(128, 128, 3)

)

train_model(
    resnet50,
    "resnet50"
)


# ==================================================
# 9. MobileNet
# ==================================================

mobilenet = MobileNet(

    weights="imagenet",

    include_top=False,

    input_shape=(128, 128, 3)

)

train_model(
    mobilenet,
    "mobilenet"
)


# ==================================================
# 10. InceptionV3
# ==================================================

inceptionv3 = InceptionV3(

    weights="imagenet",

    include_top=False,

    input_shape=(128, 128, 3)

)

train_model(
    inceptionv3,
    "inceptionv3"
)


# ==================================================
# 11. EfficientNetB0
# ==================================================

efficientnetb0 = EfficientNetB0(

    weights="imagenet",

    include_top=False,

    input_shape=(128, 128, 3)

)

train_model(
    efficientnetb0,
    "efficientnetb0"
)


# ==================================================
# 12. Complete
# ==================================================

print("\n")
print("=" * 60)
print("ALL FIVE TRANSFER LEARNING MODELS TRAINED SUCCESSFULLY!")
print("=" * 60)

print("\nModels saved in:")
print("models/")

print("\nTraining histories saved in:")
print("results/")