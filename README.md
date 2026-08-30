# Multiclass Fish Image Classification

## Project Overview

This project uses **Deep Learning and Transfer Learning** to classify fish images into five different classes.

A CNN model built from scratch is compared with five pre-trained deep learning models. The best-performing model is then integrated into a **Streamlit web application** for fish image prediction.

## Fish Classes

The project classifies the following five fish species:

* Catla
* CommonCarp
* Mori
* Rohu
* SilverCarp

## Dataset

The dataset contains **1,029 fish images** divided into training, validation, and test sets.

| Dataset    |    Images | Classes |
| ---------- | --------: | ------: |
| Training   |       719 |       5 |
| Validation |       207 |       5 |
| Test       |       103 |       5 |
| **Total**  | **1,029** |   **5** |

The original dataset also contains XML annotation files. These files are not used because this project performs **image classification**, not object detection.

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Pillow
* Streamlit

## Models Used

Six Deep Learning models were trained and compared:

1. CNN
2. VGG16
3. ResNet50
4. MobileNet
5. InceptionV3
6. EfficientNetB0

The transfer learning models use ImageNet pre-trained weights with classification layers adapted for the five fish classes.

## Data Preprocessing

The following preprocessing techniques were used:

* Image resizing to **128 × 128 pixels**
* Pixel normalization
* Data augmentation
* Horizontal flipping
* Random rotation
* Random zoom

## Model Evaluation

The models were evaluated using the test dataset with the following metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

Training history is also stored and used to generate model accuracy and loss plots.

## Model Comparison

| Model          |   Accuracy |  Precision |     Recall |   F1-Score |
| -------------- | ---------: | ---------: | ---------: | ---------: |
| **MobileNet**  | **96.12%** | **96.79%** | **96.12%** | **96.07%** |
| CNN            |     93.20% |     93.66% |     93.20% |     93.17% |
| InceptionV3    |     83.50% |     83.87% |     83.50% |     83.24% |
| VGG16          |     70.87% |     80.26% |     70.87% |     71.31% |
| ResNet50       |     29.13% |     23.22% |     29.13% |     20.57% |
| EfficientNetB0 |     15.53% |      2.41% |     15.53% |      4.18% |

## Best Model

**MobileNet** achieved the best performance on the test dataset.

* Accuracy: **96.12%**
* Precision: **96.79%**
* Recall: **96.12%**
* F1-Score: **96.07%**

Therefore, MobileNet is used in the Streamlit application for final fish image prediction.

## Training Visualizations

Training history is saved for the trained models in JSON format.

The project includes a visualization script that generates:

* Training vs. validation accuracy plots
* Training vs. validation loss plots

The generated plots are stored in:

```text
results/plots/
```

## Streamlit Application

The project includes an interactive **Streamlit web application**.

The application allows users to:

1. Upload a fish image.
2. Display the uploaded image.
3. Predict the fish class.
4. Display the prediction confidence.

The application uses the trained **MobileNet** model.

## Project Structure

```text
Multiclass-Fish-Image-Classification/
│
├── results/
│   ├── confusion_matrices/
│   ├── plots/
│   ├── *_history.json
│   └── model_comparison.csv
│
├── scripts/
│   ├── 01_train_cnn.py
│   ├── 02_train_transfer_learning.py
│   ├── 03_evaluate.py
│   ├── 04_app.py
│   └── 05_create_plots.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

> **Note:** The dataset, virtual environment, and trained model files are excluded from the repository using `.gitignore` because of their size. They are required when running the complete project locally.

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment.

### Windows

```powershell
.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Train CNN

```bash
python scripts/01_train_cnn.py
```

## Train Transfer Learning Models

```bash
python scripts/02_train_transfer_learning.py
```

This trains the following models:

* VGG16
* ResNet50
* MobileNet
* InceptionV3
* EfficientNetB0

## Evaluate Models

```bash
python scripts/03_evaluate.py
```

This generates model evaluation results, confusion matrices, and the model comparison CSV file.

## Generate Training Plots

```bash
python scripts/05_create_plots.py
```

This generates accuracy and loss plots from the saved training history files.

The generated plots are stored in:

```text
results/plots/
```

## Run Streamlit Application

```bash
python -m streamlit run scripts/04_app.py
```

Upload a fish image through the application to obtain the predicted fish class and confidence score.

## Results

Among the six tested models, **MobileNet achieved the highest test accuracy of 96.12%**.

The project demonstrates the complete Deep Learning workflow:

**Data Preprocessing → Data Augmentation → Model Training → Transfer Learning → Model Evaluation → Model Comparison → Visualization → Streamlit Deployment**

## Conclusion

This project demonstrates multiclass fish image classification using **Deep Learning and Transfer Learning**.

Six different models were trained and evaluated using multiple performance metrics. **MobileNet achieved the best overall performance with 96.12% test accuracy**.

The trained MobileNet model was integrated into a Streamlit application to provide interactive fish image classification with a confidence score.
