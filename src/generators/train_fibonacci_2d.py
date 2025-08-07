# Arquivo: src/generators/train_fibonacci_2d.py
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

# Imports corrigidos para refletir a nova estrutura de pastas
from src.models.wgan_dcgan_model import WGAN_GP
from src.utils.utils_spectrogram import generate_spectrogram, IMG_SIZE

# --- Parâmetros ---
EPOCHS = 5000
BATCH_SIZE = 64
SIGNAL_LENGTH = 4096 
N_SAMPLES = 1000
MODEL_SAVE_PATH = 'generator_fibonacci_2d.keras'
IMAGES_SAVE_DIR = 'images_fibonacci_2d'


# --- Geração de Dados (Agora gera espectrogramas) ---
def generate_fibonacci_signals(num_samples, length):
    signals = []
    for _ in range(num_samples):
        fib_sequence = [0, 1]
        while len(fib_sequence) < length:
            fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
        
        binary_signal = np.array([x % 2 for x in fib_sequence])
        signals.append(binary_signal[:length])
    
    spectrograms = [generate_spectrogram(s) for s in signals]
    # AQUI ESTÁ A CORREÇÃO: Garante que o array seja do tipo float32
    return np.array(spectrograms, dtype=np.float32)

# --- Preparação do Dataset ---
print("Gerando espectrogramas de Fibonacci para treinamento...")
X_train = generate_fibonacci_signals(N_SAMPLES, SIGNAL_LENGTH)
X_train = np.expand_dims(X_train, axis=-1)
X_train = (X_train - 0.5) / 0.5

print(f"Dataset criado. Forma: {X_train.shape}")
train_dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(N_SAMPLES).batch(BATCH_SIZE)

# --- Construção e Treinamento do Modelo ---
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
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='viridis')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig(f"{IMAGES_SAVE_DIR}/fibonacci_2d_{epoch}.png")
    plt.close()

if not os.path.exists(IMAGES_SAVE_DIR):
    os.makedirs(IMAGES_SAVE_DIR)

print("\n--- Iniciando Treinamento da WGAN-GP para Espectrogramas de Fibonacci ---")
for epoch in range(EPOCHS):
    d_loss_epoch, g_loss_epoch = [], []
    for image_batch in train_dataset:
        d_loss, g_loss = wgan.train_step(image_batch)
        d_loss_epoch.append(d_loss)
        g_loss_epoch.append(g_loss)
        
    if epoch % 100 == 0:
        avg_d_loss = np.mean(d_loss_epoch)
        avg_g_loss = np.mean(g_loss_epoch)
        print(f"Época {epoch}/{EPOCHS} \t D loss: {avg_d_loss:.4f} \t G loss: {avg_g_loss:.4f}")
        
        # Salva as imagens de amostra
        save_imgs(epoch, wgan.generator)
        
        # SALVA O MODELO PERIODICAMENTE!
        if epoch > 0: # Não salva o modelo inicial não treinado
            periodic_model_path = f"generator_fibonacci_2d_epoch_{epoch}.keras"
            wgan.generator.save(periodic_model_path)
            print(f"Modelo salvo em: {periodic_model_path}")

print("Treinamento concluído.")
wgan.generator.save(MODEL_SAVE_PATH)
print(f"Modelo salvo como {MODEL_SAVE_PATH}")