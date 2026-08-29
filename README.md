# Multiclass Fish Image Classification

## Project Overview

This project uses Deep Learning and Transfer Learning to classify fish images into five different classes.

The project compares a CNN model built from scratch with five pre-trained deep learning models.

### Fish Classes

* Catla
* CommonCarp
* Mori
* Rohu
* SilverCarp

## Dataset

The dataset contains 1,029 fish images.

| Dataset    |    Images | Classes |
| ---------- | --------: | ------: |
| Training   |       719 |       5 |
| Validation |       207 |       5 |
| Test       |       103 |       5 |
| **Total**  | **1,029** |   **5** |

The dataset is divided into training, validation, and testing sets.

XML annotation files are not used because this project performs image classification rather than object detection.

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Streamlit

## Models Used

The following models were trained and compared:

1. CNN
2. VGG16
3. ResNet50
4. MobileNet
5. InceptionV3
6. EfficientNetB0

Transfer learning models use ImageNet pre-trained weights with the original classification layers replaced by layers suitable for the five fish classes.

## Data Preprocessing

The following preprocessing techniques were used:

* Image resizing to 128 × 128 pixels
* Pixel normalization
* Data augmentation
* Horizontal flipping
* Random rotation
* Random zoom

## Model Evaluation

The models were evaluated using the test dataset with:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

### Model Comparison

| Model          |   Accuracy |  Precision |     Recall |   F1-Score |
| -------------- | ---------: | ---------: | ---------: | ---------: |
| MobileNet      | **96.12%** | **96.79%** | **96.12%** | **96.07%** |
| CNN            |     93.20% |     93.66% |     93.20% |     93.17% |
| InceptionV3    |     83.50% |     83.87% |     83.50% |     83.24% |
| VGG16          |     70.87% |     80.26% |     70.87% |     71.31% |
| ResNet50       |     29.13% |     23.22% |     29.13% |     20.57% |
| EfficientNetB0 |     15.53% |      2.41% |     15.53% |      4.18% |

### Best Model

MobileNet achieved the best test performance with:

* Accuracy: **96.12%**
* Precision: **96.79%**
* Recall: **96.12%**
* F1-score: **96.07%**

Therefore, MobileNet is used in the Streamlit application for final fish image prediction.

## Streamlit Application

The project includes a Streamlit web application.

The application allows the user to:

1. Upload a fish image.
2. Display the uploaded image.
3. Predict the fish class.
4. Display the prediction confidence.

The application uses the trained MobileNet model.

## Project Structure

```text
Multiclass-Fish-Image-Classification/
│
├── dataset/
│   └── fish_dataset/
│       ├── train/
│       ├── valid/
│       └── test/
│
├── models/
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
│   └── 04_app.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Clone the repository and create a virtual environment.

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

## Evaluate Models

```bash
python scripts/03_evaluate.py
```

## Run Streamlit Application

```bash
python -m streamlit run scripts/04_app.py
```

## Conclusion

This project demonstrates multiclass fish image classification using Deep Learning and Transfer Learning.

Among the tested models, MobileNet achieved the highest performance on the test dataset with an accuracy of **96.12%**.

The trained MobileNet model is integrated with a Streamlit application to provide real-time fish image classification and confidence prediction.
