import os
import json
import sys
import subprocess
from pathlib import Path
try:
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
    from tensorflow.keras.callbacks import ModelCheckpoint, Callback
    from tensorflow.keras.utils import to_categorical
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
except ImportError:
    # Check and install missing dependencies
    required_libs = (
        'tensorflow',
        'requests'
        'numpy',
        'matplotlib',
    )

    missing = ()
    for lib in required_libs:
        try:
            __import__(lib)
        except ImportError:
            missing.append(lib)
    print("Missing required libraries:", ", ".join(missing))
    response = input("Would you like to install them? (y/n): ").lower()
    if response == 'y':
        for lib in missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
        print("Libraries installed. Please restart the program.")
        sys.exit()
    else:
        print("Exiting program.")
        sys.exit()

CONFIG_FILE = "text_generator_config.json"
DEFAULT_DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

class TextGenerator:
    def __init__(self):
        self.config = self.load_config()
        self.model = None
        self.char_to_int = {}
        self.int_to_char = {}
        self.n_vocab = 0
        self.sequence_length = 50  # Reduced sequence length
        self.batch_size = 64

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        return {"dataset_path": None}

    def save_config(self):
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.config, f)

    def get_dataset(self):
        if self.config["dataset_path"] and os.path.exists(self.config["dataset_path"]):
            return self.load_text(self.config["dataset_path"])
            
        print("No dataset configured. Choose an option:")
        print("1. Use local text file")
        print("2. Download default dataset")
        
        while True:
            choice = input("Enter choice (1/2): ").strip()
            if choice in ['1', '2']:
                break
            print("Invalid choice. Please enter 1 or 2")

        if choice == '1':
            while True:
                path = input("Enter full path to text file: ").strip()
                if os.path.exists(path):
                    self.config["dataset_path"] = path
                    self.save_config()
                    return self.load_text(path)
                print("File not found. Try again")
        else:
            print(f"Downloading default dataset from {DEFAULT_DATA_URL}")
            response = requests.get(DEFAULT_DATA_URL)
            Path("data").mkdir(exist_ok=True)
            path = "data/default_dataset.txt"
            with open(path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            self.config["dataset_path"] = path
            self.save_config()
            return response.text

    def load_text(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    def preprocess_data(self, raw_text):
        chars = sorted(list(set(raw_text)))
        self.char_to_int = {c:i for i,c in enumerate(chars)}
        self.int_to_char = {i:c for i,c in enumerate(chars)}
        self.n_vocab = len(chars)
        
        # Optimized sequence generation using numpy
        text_as_int = np.array([self.char_to_int[c] for c in raw_text], dtype=np.int32)
        
        # Create training sequences using sliding window
        seq_length = self.sequence_length
        examples_per_epoch = len(raw_text) // (seq_length + 1)
        
        # Create training examples
        char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
        sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)
        
        # Split sequences into input/target
        def split_input_target(chunk):
            input_text = chunk[:-1]
            target_text = chunk[1:]
            return input_text, target_text
        
        dataset = sequences.map(split_input_target)
        
        # Batch and shuffle data
        self.dataset = dataset.shuffle(10000).batch(self.batch_size, drop_remainder=True)
        
        return self.dataset

    def build_model(self):
        # Fixed model architecture with sequence output
        self.model = Sequential([
            Embedding(self.n_vocab, 128, input_length=self.sequence_length),
            LSTM(256, return_sequences=True),
            Dropout(0.2),
            LSTM(256, return_sequences=True),  # Critical fix: return sequences
            Dense(self.n_vocab, activation='softmax')
        ])
        
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

    def train(self, dataset, epochs=20):
        # Simplified checkpointing
        checkpoint = ModelCheckpoint(
            "text_gen_model.keras",
            monitor='loss',
            save_best_only=True,
            save_freq='epoch'  # Save once per epoch
        )
        
        # Use TensorFlow's built-in progress bar
        history = self.model.fit(
            dataset,
            epochs=epochs,
            callbacks=[checkpoint, SimpleTrainingLogger()],
            verbose=1
        )
        
        return history

    def generate_text(self, seed=None, length=500, temperature=1.0):
        if seed is None:
            start_idx = np.random.randint(0, len(self.raw_text) - self.sequence_length)
            seed = self.raw_text[start_idx:start_idx+self.sequence_length]
            
        generated = seed
        for _ in range(length):
            x_pred = np.zeros((1, self.sequence_length))
            for t, char in enumerate(generated[-self.sequence_length:]):
                x_pred[0, t] = self.char_to_int[char]
                
            # Get only the last prediction
            preds = self.model.predict(x_pred, verbose=0)[0][-1]  # Critical fix here
            next_index = self.sample(preds, temperature)
            next_char = self.int_to_char[next_index]
            
            generated += next_char
            
        return generated

    def sample(self, preds, temperature):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        return np.random.choice(len(preds), p=preds)

class TrainingPlot(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        plt.ion()
        
    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        
        plt.clf()
        plt.plot(self.losses, label='Training loss')
        plt.plot(self.val_losses, label='Validation loss')
        plt.title('Training Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.draw()
        plt.pause(0.001)

class SimpleTrainingLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Simple text-based logging
        print(f"\nEpoch {epoch+1} - loss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f}")

def main():
    generator = TextGenerator()
    
    # Load and preprocess data
    raw_text = generator.get_dataset()
    dataset = generator.preprocess_data(raw_text)
    generator.raw_text = raw_text
    
    # Build and train model
    generator.build_model()
    print(generator.model.summary())
    
    try:
        epochs = int(input("Enter number of training epochs (default 10): ") or 10)
    except ValueError:
        print("Invalid input. Using default value")
        epochs = 10
        
    history = generator.train(dataset, epochs=epochs)
    
    # Generate sample text
    print("\nText Generation Examples:")
    temperatures = [0.5, 1.0, 1.5]
    seed = "ROMEO: "
    
    for temp in temperatures:
        print(f"\nTemperature {temp}:")
        print(generator.generate_text(seed=seed, temperature=temp))
    
    # Save final model
    generator.model.save("final_text_generator.keras")
    print("\nModel saved as 'final_text_generator.keras'")

if __name__ == "__main__":
    main()
