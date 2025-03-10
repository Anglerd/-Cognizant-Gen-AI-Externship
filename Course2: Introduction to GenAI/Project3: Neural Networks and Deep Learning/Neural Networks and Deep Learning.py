import os
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    import matplotlib.pyplot as plt
except ImportError:
    print("TensorFlow or TensorFlow Datasets is not installed.")
    print("Please install TensorFlow and TensorFlow Datasets using the following command:")
    print("pip install tensorflow tensorflow-datasets scikit-learn matplotlib")
    sys.exit(1)  # Exit the program with an error code
import zipfile

# Part 1: Building and Optimizing a CNN

# Function to download and extract the dataset
def download_and_extract_dataset(data_dir):
    dataset_url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
    dataset_zip = tf.keras.utils.get_file("cats_and_dogs_filtered.zip", origin=dataset_url, extract=False)
    with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(dataset_zip)  # Remove the downloaded ZIP file after extraction
    print(f"Dataset downloaded and extracted to {data_dir}")

# Load and preprocess the dataset
def load_dataset():
    # Prompt the user for the dataset path
    data_dir = input("Enter the path to the dataset directory (or press Enter to download it): ").strip()

    if not data_dir:
        # If no path is provided, download the dataset to a default directory
        data_dir = os.path.join(os.getcwd(), "cats_and_dogs_filtered")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        download_and_extract_dataset(data_dir)
    else:
        # Check if the provided path exists
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"The directory '{data_dir}' does not exist.")

    # Define paths
    train_dir = os.path.join(data_dir, "cats_and_dogs_filtered", "train")
    validation_dir = os.path.join(data_dir, "cats_and_dogs_filtered", "validation")

    # Check if directories exist
    if not os.path.exists(train_dir) or not os.path.exists(validation_dir):
        raise FileNotFoundError(f"Dataset directories not found. Please ensure the dataset is extracted at {data_dir}.")

    # Preprocess and augment data
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,  # Normalize pixel values
        rotation_range=20,  # Data augmentation
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    validation_datagen = ImageDataGenerator(rescale=1.0 / 255)  # Only normalize validation data

    # Create data generators
    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(150, 150), batch_size=32, class_mode="binary"
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir, target_size=(150, 150), batch_size=32, class_mode="binary"
    )

    return train_generator, validation_generator

# Build the initial CNN model
def build_cnn_model(input_shape=(150, 150, 3)):
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(256, (3, 3), activation="relu"),  # Additional Conv2D layer
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.5),  # Dropout layer to prevent overfitting
            layers.Dense(1, activation="sigmoid"),  # Binary classification
        ]
    )
    return model

# Compile the model
def compile_model(model):
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Train the model
def train_model(model, train_generator, validation_generator, epochs=10):
    history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)
    return history

# Plot training and validation accuracy/loss
def plot_history(history):
    # Plot training and validation accuracy
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    # Plot training and validation loss
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

# Part 2: Debugging Model Failures

# Analyze performance and debug issues
def debug_model(model, train_generator, validation_generator):
    # Train the model and plot history
    history = train_model(model, train_generator, validation_generator)
    plot_history(history)

    # Debugging steps
    if history.history["val_accuracy"][-1] < 0.8:  # Example condition for underfitting
        print("Model may be underfitting. Adding more layers and increasing epochs.")

        # Build a new model with additional layers
        new_model = models.Sequential()

        # Add existing layers from the original model (excluding the final Dense layer)
        for layer in model.layers[:-2]:  # Exclude the Flatten and final Dense layer
            new_model.add(layer)

        # Add new layers
        new_model.add(layers.Conv2D(256, (3, 3), activation="relu"))  # New Conv2D layer
        new_model.add(layers.MaxPooling2D((2, 2)))  # New MaxPooling layer
        new_model.add(layers.Flatten())  # Flatten the output before adding dense layers
        new_model.add(layers.Dense(1024, activation="relu"))  # New Dense layer
        new_model.add(layers.Dropout(0.5))  # Add dropout to prevent overfitting
        new_model.add(layers.Dense(1, activation="sigmoid"))  # Output layer

        # Recompile the model
        new_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        # Train the new model with more epochs
        history = train_model(new_model, train_generator, validation_generator, epochs=20)
        plot_history(history)

        return new_model

    return model

# Part 3: Evaluating Model Effectiveness

# Evaluate the model
def evaluate_model(model, validation_generator):
    # Evaluate the model on the validation set
    test_loss, test_accuracy = model.evaluate(validation_generator)
    print(f"Test Accuracy: {test_accuracy:.2f}")

    # Generate predictions
    y_pred = model.predict(validation_generator)
    y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary labels

    # Get true labels
    y_true = validation_generator.classes

    # Calculate metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print(f"AUC-ROC: {roc_auc:.2f}")

# Part 4: Creative Application

# Main function
def main():
    try:
        # Load and preprocess the dataset
        print("Loading and preprocessing the dataset...")
        train_generator, validation_generator = load_dataset()

        # Build, compile, and train the initial CNN model
        print("Building and training the initial CNN model...")
        model = build_cnn_model()
        model = compile_model(model)
        history = train_model(model, train_generator, validation_generator)

        # Plot training history
        plot_history(history)

        # Debug the model
        print("Debugging the model...")
        model = debug_model(model, train_generator, validation_generator)

        # Evaluate the model
        print("Evaluating the model...")
        evaluate_model(model, validation_generator)

        print("CNN project completed!")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the program
if __name__ == "__main__":
    main()
