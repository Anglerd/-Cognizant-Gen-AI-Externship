import os
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from sklearn.metrics import classification_report, roc_auc_score, roc_curve
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
except importError:
    print("Prerequisites not installed, please install all necessary files using the following command")
    print("pip install tensorflow scikit-learn matplotlib")
    sys.exit(1)  # Exit the program with an error code
import numpy as np

# Part 1: Prepare the Dataset

# Load a sample customer feedback dataset
def load_feedback_data():
    # Example data: Positive and negative feedback
    feedback = [
        "I love this product, it works great!",
        "Terrible experience, would not recommend.",
        "Excellent service, very satisfied.",
        "Poor quality, disappointed.",
        "Amazing product, worth every penny.",
        "Worst purchase ever, complete waste of money.",
        "Fast delivery and great packaging.",
        "Defective product, returned immediately."
    ]
    labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Positive, 0 = Negative
    return feedback, labels

# Convert text data into numerical representations (e.g., word embeddings)
def text_to_image(text_data, max_len=100, embedding_dim=128):
    # Use a pre-trained embedding model (e.g., Word2Vec, GloVe, or TF-IDF)
    # For simplicity, we'll use random embeddings here
    np.random.seed(42)
    embeddings = np.random.rand(len(text_data), max_len, embedding_dim)
    
    # Add a dummy channel dimension
    embeddings = np.expand_dims(embeddings, axis=-1)  # Shape: (batch_size, 100, 128, 1)
    
    # Repeat the single channel to create 3 channels (to match VGG16 input)
    embeddings = np.repeat(embeddings, 3, axis=-1)  # Shape: (batch_size, 100, 128, 3)
    
    return embeddings

# Load and preprocess the dataset
feedback, labels = load_feedback_data()
X = text_to_image(feedback)  # Convert text to "image-like" data
y = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Part 2: Build and Optimize the CNN

# Use a pre-trained VGG16 model for feature extraction
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(100, 128, 3))

# Freeze the base model
base_model.trainable = False

# Build the CNN model
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Add early stopping and model checkpointing
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('feedback_cnn_best.h5', monitor='val_accuracy', save_best_only=True, mode='max')

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=8,
    validation_split=0.2,
    callbacks=[early_stopping, checkpoint]
)

# Part 3: Debug and Evaluate

# Plot training/validation curves
def plot_curves(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()

plot_curves(history)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# ROC-AUC score
roc_auc = roc_auc_score(y_test, y_pred)
print(f"ROC-AUC Score: {roc_auc}")

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Save the final model
model.save('feedback_cnn_final.h5')
print("Model saved as 'feedback_cnn_final.h5'.")
