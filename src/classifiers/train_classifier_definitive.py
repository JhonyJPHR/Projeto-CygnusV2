import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
import os

from src.utils.utils_spectrogram import IMG_SIZE

# --- Parâmetros ---
N_SAMPLES_PER_CLASS = 1000
SIGNAL_LENGTH = 4096 
NUM_CLASSES = 6
CLASSIFIER_MODEL_FILE = 'universal_classifier_model_v8_definitive.keras'

AI_GENERATORS = {
    'Artificial (Fibonacci)': 'generator_fibonacci_2d_epoch_1000.keras',
    'Artificial (Pi)': 'generator_pi_2d_epoch_1300.keras',
    'Artificial (Euler)': 'generator_euler_2d_epoch_2000.keras'
}

# --- FUNÇÕES DE GERAÇÃO DE DADOS (VERSÃO FINAL E REALISTA) ---

def generate_dispersed_pulse(time_steps, freq_channels, pulse_time, pulse_width, dm):
    spectrogram = np.zeros((time_steps, freq_channels))
    freqs = np.linspace(1.2, 0.2, freq_channels)
    time_delay = dm * (freqs**-2 - 1) * (time_steps / 50)
    for i, f in enumerate(freqs):
        dispersed_time = pulse_time + time_delay[i]
        time_profile = np.exp(-0.5 * ((np.arange(time_steps) - dispersed_time) / pulse_width)**2)
        spectrogram[:, i] += time_profile
    return spectrogram

def generate_realistic_pulsar_spectrograms(num_samples, time_steps, freq_channels):
    spectrograms = []
    for _ in range(num_samples):
        noise = np.random.normal(0, 0.5, (time_steps, freq_channels))
        pulse_period = np.random.randint(100, 200)
        pulse_width = np.random.uniform(2.0, 4.0)
        dm = np.random.uniform(30, 80)
        intensity = np.random.uniform(10, 20)
        for t_start in range(pulse_period, time_steps, pulse_period):
            noise += generate_dispersed_pulse(time_steps, freq_channels, t_start, pulse_width, dm) * intensity
        final_spec = np.zeros((IMG_SIZE[0], IMG_SIZE[1]))
        h, w = min(noise.shape[0], IMG_SIZE[0]), min(noise.shape[1], IMG_SIZE[1])
        final_spec[:h, :w] = noise[:h, :w]
        spectrograms.append(final_spec)
    return np.array(spectrograms, dtype=np.float32)

def generate_realistic_frb_spectrograms(num_samples, time_steps, freq_channels):
    spectrograms = []
    for _ in range(num_samples):
        noise = np.random.normal(0, 0.5, (time_steps, freq_channels))
        pulse_time = np.random.randint(time_steps // 4, 3 * time_steps // 4)
        pulse_width = np.random.uniform(1.0, 3.0)
        dm = np.random.uniform(100, 500)
        intensity = np.random.uniform(20, 40)
        pulse = generate_dispersed_pulse(time_steps, freq_channels, pulse_time, pulse_width, dm) * intensity
        final_spec = np.zeros((IMG_SIZE[0], IMG_SIZE[1]))
        h, w = min(pulse.shape[0], IMG_SIZE[0]), min(pulse.shape[1], IMG_SIZE[1])
        final_spec[:h, :w] = (noise + pulse)[:h, :w]
        spectrograms.append(final_spec)
    return np.array(spectrograms, dtype=np.float32)

def generate_solar_flare_spectrograms(num_samples, time_steps, freq_channels):
    spectrograms = []
    for _ in range(num_samples):
        noise = np.random.randn(time_steps, freq_channels) * 0.5
        start_point = np.random.randint(0, time_steps // 2)
        decay = np.random.uniform(0.005, 0.01)
        amp = np.random.uniform(2, 4)
        time_pts = np.arange(time_steps - start_point)
        flare = amp * np.exp(-decay * time_pts)
        noise[start_point:, :] += flare[:, np.newaxis]
        final_spec = np.zeros((IMG_SIZE[0], IMG_SIZE[1]))
        h, w = min(noise.shape[0], IMG_SIZE[0]), min(noise.shape[1], IMG_SIZE[1])
        final_spec[:h, :w] = noise[:h, :w]
        spectrograms.append(final_spec)
    return np.array(spectrograms, dtype=np.float32)

def generate_ai_spectrograms(generator_filename, num_samples):
    try:
        generator = load_model(generator_filename, compile=False)
        noise = tf.random.normal([num_samples, 100]); images = generator.predict(noise, verbose=1); return images
    except Exception as e:
        print(f"Erro ao carregar o gerador 2D em {generator_filename}: {e}"); return None

# --- ARQUITETURA HÍBRIDA (sem alterações) ---
def build_classifier_hybrid(num_classes):
    model = Sequential(name="Classificador_Hibrido_CNN_LSTM_v8")
    model.add(Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)))
    model.add(Conv2D(filters=32, kernel_size=3, activation='relu', padding='same')); model.add(MaxPooling2D(pool_size=(2, 2))); model.add(Dropout(0.4))
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')); model.add(MaxPooling2D(pool_size=(2, 2))); model.add(Dropout(0.4))
    model.add(Reshape((16, 16 * 64)))
    model.add(LSTM(64)); model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']); model.summary(); return model

# --- EXECUÇÃO PRINCIPAL (VERSÃO FINAL CORRIGIDA) ---
print("Gerando datasets de espectrogramas (Versão Definitiva)...")
all_spectrograms, all_labels = [], []

# Gera dados realistas e JÁ ADICIONA A DIMENSÃO EXTRA
print("- Gerando classe realista: Pulsar")
pulsar_specs = generate_realistic_pulsar_spectrograms(N_SAMPLES_PER_CLASS, SIGNAL_LENGTH, IMG_SIZE[1])
pulsar_specs = pulsar_specs.reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1) # <--- CORREÇÃO AQUI
all_spectrograms.extend(pulsar_specs)
all_labels.extend([0] * N_SAMPLES_PER_CLASS)

print("- Gerando classe realista: FRB")
frb_specs = generate_realistic_frb_spectrograms(N_SAMPLES_PER_CLASS, SIGNAL_LENGTH, IMG_SIZE[1])
frb_specs = frb_specs.reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1) # <--- CORREÇÃO AQUI
all_spectrograms.extend(frb_specs)
all_labels.extend([1] * N_SAMPLES_PER_CLASS)

print("- Gerando classe: Solar Flare")
solar_specs = generate_solar_flare_spectrograms(N_SAMPLES_PER_CLASS, SIGNAL_LENGTH, IMG_SIZE[1])
solar_specs = solar_specs.reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1) # <--- CORREÇÃO AQUI
all_spectrograms.extend(solar_specs)
all_labels.extend([2] * N_SAMPLES_PER_CLASS)

# Carrega dados das GANs (que já estão no formato 3D correto)
ai_class_offset = 3
for i, (name, model_file) in enumerate(AI_GENERATORS.items()):
    print(f"- Carregando classe artificial: {name}")
    ai_spectrograms = generate_ai_spectrograms(model_file, N_SAMPLES_PER_CLASS)
    if ai_spectrograms is not None:
        all_spectrograms.extend(ai_spectrograms)
        all_labels.extend([ai_class_offset + i] * N_SAMPLES_PER_CLASS)

# A linha .reshape() no final não é mais necessária, pois todos os itens já estão no formato correto
X = np.array(all_spectrograms)
y = to_categorical(all_labels, num_classes=NUM_CLASSES)

print(f"\nDataset final criado com {X.shape[0]} espectrogramas de {NUM_CLASSES} classes.")
print(f"Formato do array X: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Datasets prontos para o treinamento do classificador definitivo.")

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
classifier = build_classifier_hybrid(NUM_CLASSES)

print("\n--- Treinando o Classificador Definitivo v8 (Híbrido-Realista) ---")
classifier.fit(X_train, y_train, 
             epochs=50, 
             batch_size=32, 
             validation_data=(X_test, y_test),
             callbacks=[early_stopping])

loss, accuracy = classifier.evaluate(X_test, y_test)
print(f"\nAcurácia final do Classificador Definitivo: {accuracy:.2%}")
classifier.save(CLASSIFIER_MODEL_FILE)
print(f"Modelo Definitivo salvo em: {CLASSIFIER_MODEL_FILE}")