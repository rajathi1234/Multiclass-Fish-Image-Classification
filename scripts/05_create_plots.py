import json
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


plot_folder = "results/plots"

os.makedirs(plot_folder, exist_ok=True)

models = [
    "vgg16",
    "resnet50",
    "mobilenet",
    "inceptionv3",
    "efficientnetb0"
]


for model in models:

    file_path = "results/" + model + "_history.json"

    with open(file_path, "r") as file:
        history = json.load(file)

    # Accuracy graph
    plt.figure(figsize=(8, 5))
    plt.plot(history["accuracy"], label="Training Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")

    plt.title(model.upper() + " - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()

    plt.savefig(plot_folder + "/" + model + "_accuracy.png")
    plt.close()

    # Loss graph
    plt.figure(figsize=(8, 5))
    plt.plot(history["loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")

    plt.title(model.upper() + " - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()

    plt.savefig(plot_folder + "/" + model + "_loss.png")
    plt.close()

    print(model, "plots created successfully!")


print("\nAll available plots created successfully!")