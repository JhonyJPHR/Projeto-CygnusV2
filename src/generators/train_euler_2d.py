# Arquivo: src/generators/train_euler_2d.py
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

from src.models.wgan_dcgan_model import WGAN_GP
from src.utils.utils_spectrogram import generate_spectrogram, IMG_SIZE

# --- Parâmetros ---
EPOCHS = 5000
BATCH_SIZE = 64
SIGNAL_LENGTH = 4096
N_SAMPLES = 1000
MODEL_SAVE_PATH = 'generator_euler_2d.keras'
IMAGES_SAVE_DIR = 'images_euler_2d'

# --- Geração de Dados ---
def generate_euler_signals(num_samples, length):
    # Dígitos da constante de Euler (e)
    euler_digits_str = "7182818284590452353602874713526624977572470936999595749669676277240766303535475945713821785251664274"
    euler_digits = [int(d) for d in euler_digits_str]
    
    signals = []
    for _ in range(num_samples):
        signal = []
        start_index = np.random.randint(0, len(euler_digits) - 50)
        current_digits = euler_digits[start_index:]
        
        while len(signal) < length:
            for digit in current_digits:
                if len(signal) >= length: break
                signal.append(1)
                if digit > 0:
                    zeros_to_add = min(digit, length - len(signal))
                    signal.extend([0] * zeros_to_add)
        signals.append(np.array(signal[:length]))

    spectrograms = [generate_spectrogram(s) for s in signals]
    return np.array(spectrograms, dtype=np.float32)

# --- Preparação do Dataset e Treinamento (código idêntico ao anterior) ---
print("Gerando espectrogramas de Euler para treinamento...")
X_train = generate_euler_signals(N_SAMPLES, SIGNAL_LENGTH)
X_train = np.expand_dims(X_train, axis=-1)
X_train = (X_train - 0.5) / 0.5
train_dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(N_SAMPLES).batch(BATCH_SIZE)
wgan = WGAN_GP(input_dim=(IMG_SIZE[0], IMG_SIZE[1], 1))

def save_imgs(epoch, generator):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, 100))
    gen_imgs = generator.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='viridis'); axs[i,j].axis('off')
            cnt += 1
    fig.savefig(f"{IMAGES_SAVE_DIR}/euler_2d_{epoch}.png"); plt.close()

if not os.path.exists(IMAGES_SAVE_DIR):
    os.makedirs(IMAGES_SAVE_DIR)

print("\n--- Iniciando Treinamento da WGAN-GP para Espectrogramas de Euler ---")
# (O loop de treinamento é idêntico aos outros scripts)
for epoch in range(EPOCHS):
    d_loss_epoch, g_loss_epoch = [], []
    for image_batch in train_dataset:
        d_loss, g_loss = wgan.train_step(image_batch)
        d_loss_epoch.append(d_loss); g_loss_epoch.append(g_loss)
    if epoch % 100 == 0:
        avg_d_loss = np.mean(d_loss_epoch); avg_g_loss = np.mean(g_loss_epoch)
        print(f"Época {epoch}/{EPOCHS} \t D loss: {avg_d_loss:.4f} \t G loss: {avg_g_loss:.4f}")
        save_imgs(epoch, wgan.generator)
        if epoch > 0:
            wgan.generator.save(f"{MODEL_SAVE_PATH}_epoch_{epoch}.keras")
wgan.generator.save(MODEL_SAVE_PATH)