import os
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import pickle

X = []
y = []

base_path = "C:/Users/hp/Desktop/dataset"
source_path = base_path

for child in os.listdir(source_path):
    sub_path = os.path.join(source_path, child)
    if os.path.isdir(sub_path):
        for data_file in os.listdir(sub_path):
            X_i = Image.open(os.path.join(sub_path, data_file))
            X_i = np.array(X_i.resize((120, 120))) / 255.0
            X.append(X_i)
            y.append(child)


# Convert y to numerical labels using LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Convert X and y to NumPy arrays
X = np.array(X)
y = np.array(y)

# Flatten each image into a one-dimensional array
X = X.reshape(X.shape[0], -1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)  # Check the shape of the training data
print(X_test.shape)   # Check the shape of the testing data

# Create an SVM Classifier
svm_classifier = svm.SVC(kernel='linear', C=1.0)

# Train the SVM Model
svm_classifier.fit(X_train, y_train)

# Evaluate the Model (Optional)
y_pred = svm_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Generate a classification report
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

# Load a single image
image_path = "C:/Users/hp/Desktop/74-2.jpg"  # Replace with the actual path to your image
single_image = Image.open(image_path)

# Preprocess the single image
single_image = np.array(single_image.resize((120, 120))) / 255.0

# Flatten the single image
single_image = single_image.reshape(1, -1)

# Use the trained SVM classifier to make a prediction
prediction = svm_classifier.predict(single_image)

print("predicted class",prediction)

plt.imshow(single_image.reshape(120, 120, 3))  # Reshape to original image dimensions (assuming RGB)
plt.show()

pickle.dump(svm_classifier, open('model.pkl','wb'))