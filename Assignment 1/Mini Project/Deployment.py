
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import numpy as np

# Constants
time = 10
offset = 100
folders = ["LAYING", "SITTING", "STANDING", "WALKING", "WALKING_DOWNSTAIRS", "WALKING_UPSTAIRS"]
classes = {"WALKING": 1, "WALKING_UPSTAIRS": 2, "WALKING_DOWNSTAIRS": 3, "SITTING": 4, "STANDING": 5, "LAYING": 6}

combined_dir = os.path.join("Combined")

# Train Dataset
X_train = []
y_train = []
dataset_dir_train = os.path.join(combined_dir, "Train")

for folder in folders:
    files = os.listdir(os.path.join(dataset_dir_train, folder))

    for file in files:
        df = pd.read_csv(os.path.join(dataset_dir_train, folder, file), sep=",", header=0)
        df = df[offset:offset + time * 50]
        X_train.append(df.values.flatten())  # Flatten the features
        y_train.append(classes[folder])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Test Dataset
X_test = []
y_test = []
dataset_dir_test = r"C:\Users\rutuj\Downloads\human+activity+recognition+using+smartphones\Activities"

for folder in folders:
    files = os.listdir(os.path.join(dataset_dir_test, folder))
    for file in files:
        df = pd.read_csv(os.path.join(dataset_dir_test, folder, file), sep=",", header=0, on_bad_lines='skip')
        df = df[offset:offset + time * 50]
        
        # Check for NaN values in df
        if df.isnull().values.any():
            print(f"NaN values found in folder: {folder}, file: {file}")
        
        X_test.append(df.values.flatten())  # Flatten the features
        y_test.append(classes[folder])

# Convert to NumPy array after processing
X_test = np.array(X_test)
y_test = np.array(y_test)

# # Create a Decision Tree model with adjusted max_depth
# max_depth_value = 5  # Change this value to your desired depth
# dt_model = DecisionTreeClassifier(random_state=4, max_depth=max_depth_value)

# # Train the Decision Tree model
# dt_model.fit(X_train, y_train)

# # Predict on the testing set
# y_pred = dt_model.predict(X_test)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# conf_matrix = confusion_matrix(y_test, y_pred)

# # Print the results
# print("Accuracy:", accuracy)
# print("Confusion Matrix:")
# print(conf_matrix)

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assuming X_train, y_train, X_test, and y_test are your training and testing sets
# Customize the hyperparameters based on your needs
custom_tree = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features=None,  # You can specify a number or 'sqrt' or 'log2'
    criterion='gini'    # or 'entropy'
)

# Train the model
custom_tree.fit(X_train, y_train)

# Make predictions on the test set
y_pred = custom_tree.predict(X_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)










