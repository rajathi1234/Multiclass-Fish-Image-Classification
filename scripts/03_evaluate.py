import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import load_model
from keras.utils import load_img, img_to_array

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ==========================================
# 1. Paths
# ==========================================

TEST_PATH = "dataset/fish_dataset/test"
MODEL_PATH = "models"
RESULT_PATH = "results"


# ==========================================
# 2. Classes
# ==========================================

classes = [
    "Catla",
    "CommonCarp",
    "Mori",
    "Rohu",
    "SilverCarp"
]

class_to_index = {
    name: index for index, name in enumerate(classes)
}


# ==========================================
# 3. Load test data
# ==========================================

def load_test_data(folder):

    images = []
    labels = []

    for filename in os.listdir(folder):

        if filename.lower().endswith(".jpg"):

            class_name = filename.split("_")[0]

            if class_name in class_to_index:

                image_path = os.path.join(
                    folder,
                    filename
                )

                image = load_img(
                    image_path,
                    target_size=(128, 128)
                )

                image = img_to_array(image)

                images.append(image)

                labels.append(
                    class_to_index[class_name]
                )

    return np.array(images), np.array(labels)


print("Loading test data...")

X_test, y_test = load_test_data(TEST_PATH)

X_test = X_test / 255.0

print("Test images:", len(X_test))


# ==========================================
# 4. Model names
# ==========================================

model_names = [
    "cnn",
    "vgg16",
    "resnet50",
    "mobilenet",
    "inceptionv3",
    "efficientnetb0"
]


# ==========================================
# 5. Create result folder
# ==========================================

os.makedirs(
    f"{RESULT_PATH}/confusion_matrices",
    exist_ok=True
)


results = []


# ==========================================
# 6. Evaluate all models
# ==========================================

for model_name in model_names:

    print("\n")
    print("=" * 60)
    print("Evaluating:", model_name)
    print("=" * 60)

    model_file = (
        f"{MODEL_PATH}/"
        f"{model_name}_fish_classifier.keras"
    )

    model = load_model(model_file)

    predictions = model.predict(
        X_test,
        verbose=0
    )

    y_pred = np.argmax(
        predictions,
        axis=1
    )


    # --------------------------------------
    # Calculate metrics
    # --------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )


    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")


    # --------------------------------------
    # Store results
    # --------------------------------------

    results.append({

        "Model": model_name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1_Score": f1

    })


    # --------------------------------------
    # Confusion matrix
    # --------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )


    plt.figure(figsize=(7, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=classes,
        yticklabels=classes
    )

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.title(
        f"{model_name} Confusion Matrix"
    )

    plt.tight_layout()


    confusion_file = (
        f"{RESULT_PATH}/"
        f"confusion_matrices/"
        f"{model_name}_confusion_matrix.png"
    )

    plt.savefig(confusion_file)

    plt.close()


# ==========================================
# 7. Create comparison table
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1_Score",
    ascending=False
)


# ==========================================
# 8. Save comparison
# ==========================================

comparison_file = (
    f"{RESULT_PATH}/model_comparison.csv"
)

results_df.to_csv(
    comparison_file,
    index=False
)


# ==========================================
# 9. Display final comparison
# ==========================================

print("\n")
print("=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(index=False)
)

print("\nModel comparison saved successfully!")

print("\nEvaluation completed successfully!")