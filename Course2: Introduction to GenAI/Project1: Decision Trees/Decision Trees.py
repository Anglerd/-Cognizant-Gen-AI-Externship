import pandas as pd
from sklearn.model_selection import train_test_split

# Create the dataset
data = {
    "Temperature": [25, 18, 30, 22, 28, 15],
    "Humidity": [70, 85, 60, 90, 65, 95],
    "Wind Speed": [10, 15, 5, 20, 8, 25],
    "Precipitation": [0, 5, 0, 10, 0, 15],
    "Weather Condition": ["Sunny", "Rainy", "Sunny", "Rainy", "Sunny", "Rainy"]
}

df = pd.DataFrame(data)

# Split features and target
X = df.drop("Weather Condition", axis=1)
y = df["Weather Condition"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.tree import DecisionTreeClassifier

# Create the decision tree model
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

# Predict on training and testing sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Testing Accuracy: {test_accuracy:.2f}")

# Adjust the tree depth to prevent overfitting
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Re-evaluate accuracy
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy (Adjusted): {train_accuracy:.2f}")
print(f"Testing Accuracy (Adjusted): {test_accuracy:.2f}")

# Predict on the test set
y_test_pred = model.predict(X_test)
print("Test Set Predictions:", y_test_pred)

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Calculate accuracy
accuracy = accuracy_score(y_test, y_test_pred)
print(f"Accuracy: {accuracy:.2f}")

# Calculate precision and recall
precision = precision_score(y_test, y_test_pred, pos_label="Rainy")
recall = recall_score(y_test, y_test_pred, pos_label="Rainy")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:")
print(conf_matrix)
