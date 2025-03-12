import os
import sys
import subprocess
import urllib.request
import zipfile
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    import numpy as np
    import matplotlib.pyplot as plt
    import configparser
except ImportError:
    # Check and install missing dependencies
    required_libs = [
        ('tensorflow', 'tensorflow'),
        ('configparser', 'configparser'),
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib')
    ]

    missing = []
    for lib, package in required_libs:
        try:
            __import__(lib)
        except ImportError:
            missing.append(package)
    print("Missing required libraries:", ", ".join(missing))
    response = input("Would you like to install them? (y/n): ").lower()
    if response == 'y':
        for package in missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print("Libraries installed. Please restart the program.")
        sys.exit()
    else:
        print("Exiting program.")
        sys.exit()

# Configuration setup
config = configparser.ConfigParser()
config_file = 'config.ini'
dataset_url = 'https://physiologie.unibe.ch/supplementals/delaunay.zip'
default_dataset_name = 'delaunay'

def validate_dir(path):
    if os.path.isdir(path):
        if any(file.lower().endswith(('.png', '.jpg', '.jpeg')) for file in os.listdir(path)):
            return True
    return False

def setup_dataset():
    if not os.path.exists(config_file):
        print("No configuration found. Let's set up the dataset.")
        while True:
            choice = input("Choose an option:\n1. Enter dataset path\n2. Download default dataset\n> ")
            if choice == '1':
                path = input("Enter full path to dataset directory: ").strip()
                if validate_dir(path):
                    config['DEFAULT'] = {'dataset_path': path}
                    with open(config_file, 'w') as f:
                        config.write(f)
                    return path
                else:
                    print("Invalid directory or no images found. Try again.")
            elif choice == '2':
                try:
                    print("Downloading dataset...")
                    zip_path, _ = urllib.request.urlretrieve(dataset_url)
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        extract_path = os.path.join(os.getcwd(), default_dataset_name)
                        zip_ref.extractall(extract_path)
                    config['DEFAULT'] = {'dataset_path': extract_path}
                    with open(config_file, 'w') as f:
                        config.write(f)
                    return extract_path
                except Exception as e:
                    print(f"Download failed: {str(e)}")
                    sys.exit(1)
            else:
                print("Invalid choice. Try again.")

# Load or create config
dataset_path = None
if os.path.exists(config_file):
    config.read(config_file)
    dataset_path = config['DEFAULT'].get('dataset_path')
    if not validate_dir(dataset_path):
        dataset_path = setup_dataset()
else:
    dataset_path = setup_dataset()

print(f"Using dataset from: {dataset_path}")

# GAN implementation
IMG_SIZE = 128
BATCH_SIZE = 32
LATENT_DIM = 256
EPOCHS = 100

def build_generator():
    generator = keras.Sequential([
        layers.Dense(8 * 8 * 256, input_dim=LATENT_DIM),
        layers.Reshape((8, 8, 256)),
        layers.Conv2DTranspose(128, (4,4), strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(128, (4,4), strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(64, (4,4), strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        # Additional layer to reach 128x128 resolution
        layers.Conv2DTranspose(32, (4,4), strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(3, (7,7), padding='same', activation='sigmoid')
    ])
    return generator

def build_discriminator():
    discriminator = keras.Sequential([
        layers.Conv2D(64, (3,3), strides=2, padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.4),
        layers.Conv2D(128, (3,3), strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.4),
        layers.Conv2D(256, (3,3), strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.4),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    return discriminator

class GAN(keras.Model):
    def __init__(self, generator, discriminator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = LATENT_DIM
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

    def compile(self, g_opt, d_opt, loss_fn):
        super().compile()
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        
        # Train discriminator
        noise = tf.random.normal(shape=(batch_size, self.latent_dim))
        generated_images = self.generator(noise, training=False)
        with tf.GradientTape() as d_tape:
            real_preds = self.discriminator(real_images, training=True)
            fake_preds = self.discriminator(generated_images, training=True)
            d_loss = self.loss_fn(tf.ones_like(real_preds), real_preds) * 0.5
            d_loss += self.loss_fn(tf.zeros_like(fake_preds), fake_preds) * 0.5
        
        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))
        
        # Train generator
        noise = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as g_tape:
            generated_images = self.generator(noise, training=True)
            fake_preds = self.discriminator(generated_images, training=False)
            g_loss = self.loss_fn(tf.ones_like(fake_preds), fake_preds)
        
        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(g_grads, self.generator.trainable_variables))
        
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result()
        }

# Data loading and preprocessing
def load_dataset():
    return tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        label_mode=None,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True
    ).map(lambda x: x / 255.0)

def train():
    dataset = load_dataset()
    generator = build_generator()
    discriminator = build_discriminator()
    
    gan = GAN(generator=generator, discriminator=discriminator)
    gan.compile(
        g_opt=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        d_opt=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        loss_fn=keras.losses.BinaryCrossentropy()
    )
    
    # Create output directory
    os.makedirs("generated_images", exist_ok=True)
    
    # Training loop
    history = {'g_loss': [], 'd_loss': []}
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        for batch in dataset:
            metrics = gan.train_step(batch)
        
        history['g_loss'].append(metrics['g_loss'].numpy())
        history['d_loss'].append(metrics['d_loss'].numpy())
        
        # Generate sample images
        if (epoch + 1) % 10 == 0:
            noise = tf.random.normal(shape=(5, LATENT_DIM))
            generated = generator(noise)
            plt.figure(figsize=(10, 2))
            for i in range(5):
                plt.subplot(1, 5, i+1)
                plt.imshow(generated[i])
                plt.axis('off')
            plt.savefig(f"generated_images/epoch_{epoch+1}.png")
            plt.close()
    
    # Save final model and plot training history
    generator.save("abstract_art_generator.h5")
    plt.plot(history['g_loss'], label='Generator Loss')
    plt.plot(history['d_loss'], label='Discriminator Loss')
    plt.legend()
    plt.savefig("training_history.png")
    plt.close()

if __name__ == "__main__":
    train()
    print("Training complete! Check generated_images directory for outputs.")
