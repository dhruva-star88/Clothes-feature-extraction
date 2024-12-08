# Feature Extraction with Machine Learning

This repository demonstrates an end-to-end pipeline for image classification using machine learning. The project involves:
1. **Data Preparation**: Organizing and splitting images into training and validation datasets by category.
2. **Feature Extraction**: Using `img2vec` to extract feature vectors from images.
3. **Model Training**: Training a classifier (e.g., Random Forest) on the extracted features.
4. **Performance Evaluation**: Testing the model's accuracy on the validation dataset.

---

## Features
- Categorization of images into **Topwear**, **Bottomwear**, **Footwear**, and **Accessories**.
- Train/Validation dataset splitting to ensure fair performance evaluation.
- Feature extraction using `img2vec` with pretrained `ResNet-18`.
- Model training and validation using `RandomForestClassifier`.

---

## Directory Structure
```
├── data/                 # Train/Validation datasets
│   ├── train/
│   └── val/
├── model.p               # Saved trained model
├── inference.py # Script for extracting features using img2vec
├── main.py        # Script for training and testing the model
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## Setup and Usage

### Prerequisites
- Python 3.8 or above
- Required libraries listed in `requirements.txt`

### Install Dependencies
```bash
pip install -r requirements.txt
```

## How to run
#### Note: Add your image path to ```image_path``` variable
```bash
python inference.py
```


