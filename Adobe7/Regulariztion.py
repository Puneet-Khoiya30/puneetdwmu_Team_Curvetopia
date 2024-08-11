import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Utility functions
def load_image(image_path):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def display_image(image, size=(5, 5)):
    plt.figure(figsize=size)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def convert_to_binary(image_rgb):
    gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return binary_image

def get_largest_contour(image_rgb):
    binary_image = convert_to_binary(image_rgb)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=cv2.contourArea)

def overlay_contours(image, contours, index=-1, color=(255, 0, 0), thickness=2):
    image_with_contours = image.copy()
    cv2.drawContours(image_with_contours, contours, index, color, thickness)
    display_image(image_with_contours)

def chain_code_histogram(image_rgb):
    contour = get_largest_contour(image_rgb)
    
    direction_map = {
        (1, 0): 0, (1, -1): 1, (0, -1): 2, (-1, -1): 3,
        (-1, 0): 4, (-1, 1): 5, (0, 1): 6, (1, 1): 7
    }
    
    histogram = np.zeros(8)
    for i in range(len(contour) - 1):
        point1 = contour[i][0]
        point2 = contour[i + 1][0]
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        code = direction_map.get((dx, dy), -1)
        if code != -1:
            histogram[code] += 1

    return histogram / histogram.sum() if histogram.sum() != 0 else histogram

def build_dataframe(data_directory):
    label_mapping = {
        'circle': 0,
        'square': 1,
        'star': 2,
        'triangle': 3
    }
    
    data = []
    for label_name, label_id in label_mapping.items():
        folder_path = os.path.join(data_directory, label_name)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if file_path.lower().endswith('.png'):
                data.append([file_path, label_id])
    
    return pd.DataFrame(data, columns=['image_path', 'label'])

def compute_geometric_features(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    x, y, width, height = cv2.boundingRect(contour)
    aspect_ratio = width / height
    extent = area / (width * height)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0
    epsilon = 0.01 * perimeter
    approx_poly = cv2.approxPolyDP(contour, epsilon, True)
    corners = len(approx_poly)
    
    return np.array([area, perimeter, aspect_ratio, extent, solidity, circularity, corners])

def extract_features_from_images(image_paths):
    num_images = len(image_paths)
    num_geometric_features = 7
    feature_matrix = np.zeros((num_images, 8 + num_geometric_features))
    
    for idx in tqdm(range(num_images)):
        path = image_paths[idx]
        image = load_image(path)
        histogram_features = chain_code_histogram(image)
        largest_contour = get_largest_contour(image)
        geo_features = compute_geometric_features(largest_contour)
        feature_matrix[idx] = np.concatenate([histogram_features, geo_features])
    
    return feature_matrix

def classify_image(model, image):
    class_names = ['circle', 'square', 'star', 'triangle']
    contour = get_largest_contour(image)
    histogram_features = chain_code_histogram(image)
    geometric_features = compute_geometric_features(contour)
    combined_features = np.concatenate([histogram_features, geometric_features]).reshape(1, -1)
    prediction = model.predict(combined_features)[0]
    return class_names[prediction]

def identify_shapes(model, image, width_range, height_range):
    binary_image = convert_to_binary(image)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=4)
    centroids = centroids.astype(int)
    image_with_shapes = image.copy()
    for i in range(1, num_labels):
        x, y, width, height, _ = stats[i]
        (cx, cy) = centroids[i]
        if width_range[0] < width < width_range[1] and height_range[0] < height < height_range[1]:
            shape_image = image[y:y+height, x:x+width]
            shape_label = classify_image(model, shape_image)
            cv2.rectangle(image_with_shapes, (x, y), (x+width, y+height), (255, 0, 0), 2)
            cv2.putText(image_with_shapes, shape_label, (x, y - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (255, 0, 0))
    display_image(image_with_shapes, size=(10, 10))

# Main code execution
data_dir = 'Enter the directory'
dataframe = build_dataframe(data_dir)
features = extract_features_from_images(dataframe['image_path'])
labels = dataframe['label']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

model_dict = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'XGBoost': XGBClassifier(eval_metric='logloss')
}

for model_name, model in model_dict.items():
    model.fit(X_train, y_train)
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    print(f"{model_name} - Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")

test_image_path = "C:/Users/Puneet/Downloads/img2.png"
test_image = load_image(test_image_path)
display_image(test_image)

identify_shapes(model, test_image, width_range=(1, 600), height_range=(1, 600))
