import os
import cv2
import glob
import numpy as np

# Feature extraction
from skimage.feature import hog

# Machine learning
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score

# 1. GATHER IMAGES & BUILD DATASET

DATASET_DIR = "./reference_images" #PLEASE UPDATE PATH TO REFERENCE_IMAGES

# Since we have folders like ["plastic", "paper", "metal", "organic_waste", ...]
categories = [
    cat for cat in sorted(os.listdir(DATASET_DIR)) # Lists all subdirectories inside dataset folder
    if not cat.startswith('.')  # Skips hidden folders/files like .DS_Store
]

print("Detected categories:", categories)

X = []  # Holds feature vectors
y = []  # Holds labels (as indices)

for label_index, category in enumerate(categories):
    category_path = os.path.join(DATASET_DIR, category)
    
    # Collect all image files
    image_files = (glob.glob(os.path.join(category_path, "*.jpg")) +
                   glob.glob(os.path.join(category_path, "*.png")) +
                   glob.glob(os.path.join(category_path, "*.jpeg")))
    
    for image_file in image_files:
        img = cv2.imread(image_file)
        if img is None:
            continue
        
        # Convert to grayscale for HOG
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Resize to a consistent size for feature extraction
        gray = cv2.resize(gray, (128, 128))
        
        # Extract HOG features
        hog_features = hog(
            gray,
            orientations=9, # Bins gradient directions into 9 discrete angles
            pixels_per_cell=(8, 8), # Sets each cell size to an 8x8 grid of pixels
            cells_per_block=(2, 2), # Aggregates 2x2 cells for normalization
            block_norm='L2-Hys', # L2-Hys normalization to enhance contrast invariance
            transform_sqrt=True # Stabalizes gradients in very bright/dark regions
        )
        
        X.append(hog_features)
        y.append(label_index)

X = np.array(X)
y = np.array(y)

print(f"Total images: {len(X)}")
print(f"Feature vector shape: {X.shape}")

# 2. SPLIT DATA INTO TRAIN & TEST

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# X is the feature vectors
# y is the class labels that correspond to the feature vectors
# test_size=0.2 allocates 20% of the data to the test set 
# random_state=42 fixes the seed to a random number generator 
# stratify=y ensures each category is proportionally represented in test and training set

print(f"Training samples: {len(y_train)}")
print(f"Testing samples:  {len(y_test)}")

# 3. TRAIN A CLASSICAL ML CLASSIFIER

# Create an SVM pipeline (scaling + SVC)
classifier = make_pipeline(
    StandardScaler(), #StandardScaler standardizes the features to have zero mean and unit variance
    SVC(kernel='rbf', C=1.0, gamma='scale')
)

classifier.fit(X_train, y_train)

# 4. EVALUATE THE CLASSIFIER

# Predictions on test set
y_test_pred = classifier.predict(X_test)

print("\nTest Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nClassification Report:\n",
      classification_report(y_test, y_test_pred, target_names=categories))

# 5. SIMPLE PREDICTION FUNCTION

def predict_garbage(image_path, model, hog_size=(128, 128)):
    """Loads a single image, extracts HOG features, returns predicted category."""
    img = cv2.imread(image_path)
    if img is None:
        return "Cannot read image"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, hog_size)

    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        transform_sqrt=True
    ).reshape(1, -1)

    label_index = model.predict(hog_features)[0]
    return categories[label_index]

# Example usage with a new image
test_image_path = "./image.jpg" #Replace with any image
pred_label = predict_garbage(test_image_path, classifier)
print(f"Prediction for '{test_image_path}': {pred_label}")
