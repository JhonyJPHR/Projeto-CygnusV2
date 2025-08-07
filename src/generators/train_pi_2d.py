# Arquivo: src/generators/train_pi_2d.py
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

# Importando os módulos do nosso projeto
from src.models.wgan_dcgan_model import WGAN_GP
from src.utils.utils_spectrogram import generate_spectrogram, IMG_SIZE

# --- Parâmetros ---
EPOCHS = 5000 # Pode parar antes se a qualidade for boa
BATCH_SIZE = 64
SIGNAL_LENGTH = 4096
N_SAMPLES = 1000
MODEL_SAVE_PATH = 'generator_pi_2d.keras'
IMAGES_SAVE_DIR = 'images_pi_2d'

# --- Geração de Dados ---
def generate_pi_signals(num_samples, length):
    # Uma longa sequência de dígitos de Pi
    pi_digits_str = "1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"
    pi_digits = [int(d) for d in pi_digits_str]
    
    signals = []
    for _ in range(num_samples):
        # Gera um sinal baseado na codificação de frequência (dígito = duração do silêncio)
        signal = []
        start_index = np.random.randint(0, len(pi_digits) - 50) # Pega um trecho aleatório
        current_digits = pi_digits[start_index:]
        
        while len(signal) < length:
            for digit in current_digits:
                if len(signal) >= length: break
                signal.append(1) # Pulso
                # Adiciona um número de zeros igual ao dígito
                if digit > 0:
                    zeros_to_add = min(digit, length - len(signal))
                    signal.extend([0] * zeros_to_add)
        signals.append(np.array(signal[:length]))

    spectrograms = [generate_spectrogram(s) for s in signals]
    return np.array(spectrograms, dtype=np.float32)

# --- Preparação do Dataset e Treinamento (código similar ao de Fibonacci) ---
print("Gerando espectrogramas de Pi para treinamento...")
X_train = generate_pi_signals(N_SAMPLES, SIGNAL_LENGTH)
X_train = np.expand_dims(X_train, axis=-1)
X_train = (X_train - 0.5) / 0.5

print(f"Dataset criado. Forma: {X_train.shape}")
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
    fig.savefig(f"{IMAGES_SAVE_DIR}/pi_2d_{epoch}.png"); plt.close()

if not os.path.exists(IMAGES_SAVE_DIR):
    os.makedirs(IMAGES_SAVE_DIR)

print("\n--- Iniciando Treinamento da WGAN-GP para Espectrogramas de Pi ---")
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

print("Treinamento concluído.")
wgan.generator.save(MODEL_SAVE_PATH)
print(f"Modelo salvo como {MODEL_SAVE_PATH}")