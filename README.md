# Garbage Classification using Histogram of Oriented Gradients (HOG)

## Overview
This project implements an image-based garbage classification system using **Histogram of Oriented Gradients (HOG)** for feature extraction and **Support Vector Machines (SVM)** for classification. The system is designed to identify and categorize different types of waste, such as **battery, biological, cardboard, clothes, glass, metal, paper, plastic, shoes, and trash**. 

## Features
- Uses **HOG** for extracting shape and texture features from images.
- **SVM with an RBF Kernel** to classify garbage images.
- Implements **dataset preprocessing, training, and evaluation**.
- **Handles imbalanced datasets** using stratified data splitting.
- Provides a **prediction function** to classify new images.

---

## Dataset
The dataset consists of **15,303 images** across **10 categories**:
| Category      | Number of Images |
|--------------|----------------|
| Battery      | 1,045 |
| Biological   | 1,185 |
| Cardboard    | 991 |
| Clothes      | 5,325 |
| Glass        | 1,000 |
| Metal        | 890 |
| Paper        | 1,150 |
| Plastic      | 995 |
| Shoes        | 2,000 |
| Trash        | 750 |

Each category folder contains images in **JPEG, PNG, or JPG format**.

---

## Project Structure
```
├── reference_images/     # Dataset (must be added manually)
│   ├── battery/
│   ├── biological/
│   ├── cardboard/
│   ├── ...
├── GarbageClassification.py        # Main Python script for training and prediction
├── Garbage Classification using Histogram of Oriented Gradients (HOG).pdf    # Project report detailing methodology
├── README.md             # This file
```

---

## Installation & Dependencies
Ensure you have Python installed, then install dependencies:

```bash
pip install numpy opencv-python scikit-learn scikit-image matplotlib seaborn
```

---

## How It Works
### 1. Image Preprocessing
- Converts images to **grayscale**.
- Resizes them to **128x128 pixels**.
- Extracts **HOG features** for classification.

### 2. Data Splitting
- Uses **80%** of data for training, **20%** for testing.
- **Stratified sampling** ensures balance across categories.

### 3. Model Training
- **Support Vector Machine (SVM)** with **RBF kernel** is trained using extracted features.
- **StandardScaler** normalizes the features.

### 4. Evaluation
- Computes **Accuracy, Precision, Recall, F1-score**.
- Generates **Confusion Matrix** to visualize misclassifications.

### 5. Prediction on New Images
Use the trained model to classify an image:

```python
from GarbageCollection import predict_garbage

image_path = "./image.jpg"  # Replace with your image
category = predict_garbage(image_path, classifier)
print(f"Predicted Category: {category}")
```

---

## Model Performance
- **Overall Accuracy: ~71%**
- **Best performing categories:** Clothes, Shoes, Trash.
- **Common misclassification:** Plastic vs. Paper, Metal vs. Biological Waste.

---

## Future Improvements
- **Integrate Color Histograms** for better distinction.
- **Hybrid Feature Extraction (HOG + SIFT)** to improve accuracy.
- **Ensemble Learning** to combine multiple classifiers.

---

## Author
**Tanya Budhrani**  
*Hong Kong Polytechnic University*
