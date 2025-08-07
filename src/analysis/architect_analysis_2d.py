import tensorflow as tf
import numpy as np
import argparse
import os
import csv
from datetime import datetime

from src.utils.utils_spectrogram import generate_spectrogram, IMG_SIZE
from sklearn.cluster import KMeans
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function

# --- CONFIGURAÇÃO ---
# In src/analysis/architect_analysis_2d.py
CLASSIFIER_MODEL_FILE = 'universal_classifier_model_v8_definitive.keras'
CLASS_NAMES = ['Pulsar', 'FRB', 'Solar Flare', 'Artificial (Fibonacci)', 'Artificial (Pi)', 'Artificial (Euler)']
ARTIFICIAL_CLASSES = ['Artificial (Fibonacci)', 'Artificial (Pi)', 'Artificial (Euler)']
COSMIC_LIBRARY_FILE = 'cosmic_library.csv'

# --- MÓDULO DE REGRESSÃO SIMBÓLICA (sem alterações) ---
def _protected_exp(x):
    with np.errstate(over='ignore'):
        return np.exp(np.clip(x, -700, 700))
exp_func = make_function(function=_protected_exp, name='exp', arity=1)

def decode_signal_to_sequence(signal_1d, k_clusters=16, word_length=8):
    # ... (código da função sem alterações)
    print("\n--- [Fase 2.1] Decodificando sinal para sequência numérica...")
    n_bits = len(signal_1d)
    trimmed_length = n_bits - (n_bits % word_length)
    if trimmed_length < word_length:
        print("Sinal muito curto para decodificação.")
        return None
    windows = signal_1d[:trimmed_length].reshape(-1, word_length)
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10).fit(windows)
    def to_int(word): return int("".join(map(str, np.round(word).astype(int))), 2)
    translation_map = {i: to_int(word) for i, word in enumerate(kmeans.cluster_centers_)}
    cluster_labels = kmeans.predict(windows)
    numeric_sequence = np.array([translation_map.get(label, 0) for label in cluster_labels])
    print(f"Sinal decodificado em uma sequência de {len(numeric_sequence)} números.")
    return numeric_sequence

def find_formula_specialist(n, sequence, brain, feature_names=['n'], generations=15, population_size=2000):
    # ... (código da função sem alterações)
    est = SymbolicRegressor(population_size=population_size, generations=generations,
                            stopping_criteria=1e-5, verbose=1,
                            feature_names=feature_names, function_set=brain,
                            const_range=(-5., 5.), random_state=42)
    est.fit(n, sequence)
    return est

# --- NOVA FUNÇÃO DE LOGGING ---
def log_results_to_library(log_data):
    """Adiciona os resultados da análise ao arquivo CSV da Biblioteca Cósmica."""
    file_exists = os.path.isfile(COSMIC_LIBRARY_FILE)
    
    with open(COSMIC_LIBRARY_FILE, 'a', newline='') as csvfile:
        fieldnames = ['timestamp', 'signal_source', 'classification', 'confidence', 
                      'trend_formula', 'oscillation_formula', 'final_error']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            
        writer.writerow(log_data)
    print(f"\n✅ Análise registrada com sucesso na Biblioteca Cósmica: {COSMIC_LIBRARY_FILE}")

# --- ARQUITETO PRINCIPAL (MODIFICADO PARA LOGAR) ---
def run_architect_analysis(signal_path):
    print("\n" + "="*60)
    print("🏛️  INICIANDO ARQUITETO DO CYGNUS (COM BIBLIOTECA) 🏛️")
    print("="*60)
    
    # Prepara o dicionário para guardar os resultados
    results = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'signal_source': os.path.basename(signal_path),
        'classification': 'N/A', 'confidence': 0.0,
        'trend_formula': 'N/A', 'oscillation_formula': 'N/A', 'final_error': 'N/A'
    }

    # --- FASE 1: CLASSIFICAÇÃO ---
    print(f"\n--- [Fase 1] Executando Classificador Robusto ({CLASSIFIER_MODEL_FILE})...")
    classifier = tf.keras.models.load_model(CLASSIFIER_MODEL_FILE)
    TARGET_LENGTH = 4096

    try:
        with open(signal_path, 'r') as f:
            content = f.read().strip()
        signal_1d = np.array([int(char) for char in content], dtype=np.float32)

        if len(signal_1d) > TARGET_LENGTH:
            print(f"Aviso: Sinal com {len(signal_1d)} amostras, cortando para {TARGET_LENGTH}.")
            signal_1d = signal_1d[:TARGET_LENGTH]
        elif len(signal_1d) < TARGET_LENGTH:
            print(f"Aviso: Sinal com {len(signal_1d)} amostras, preenchendo com zeros para {TARGET_LENGTH}.")
            padding = np.zeros(TARGET_LENGTH - len(signal_1d), dtype=np.float32)
            signal_1d = np.concatenate([signal_1d, padding])
    except Exception as e:
        print(f"❌ Erro ao carregar o arquivo de sinal: {e}"); return
        
    spectrogram = generate_spectrogram(signal_1d)
    input_data = spectrogram.reshape(1, *spectrogram.shape, 1)
    
    prediction_probs = classifier.predict(input_data, verbose=0)[0]
    veredito_index = np.argmax(prediction_probs)
    veredito_nome = CLASS_NAMES[veredito_index]
    confianca = np.max(prediction_probs) * 100
    
    results['classification'] = veredito_nome
    results['confidence'] = f"{confianca:.2f}%"
    print(f"\nVeredito da Fase 1: O sinal é classificado como '{veredito_nome}' com {confianca:.2f}% de confiança.")
    
    # --- FASE 2: INVESTIGAÇÃO COLABORATIVA ---
    if veredito_nome in ARTIFICIAL_CLASSES:
        print("\nSinal classificado como ARTIFICIAL. Iniciando análise profunda...")
        numeric_sequence = decode_signal_to_sequence(signal_1d)
        if numeric_sequence is not None:
            n = np.arange(len(numeric_sequence)).reshape(-1, 1)
            
            print("\n--- [Fase 2.2] 🧠 TrendHunter está a analisar a tendência...")
            trend_brain = ('add', 'sub', 'mul', 'div', 'log', exp_func)
            trend_model = find_formula_specialist(n, numeric_sequence, trend_brain)
            results['trend_formula'] = str(trend_model._program)
            
            trend_prediction = trend_model.predict(n)
            trend_prediction = np.nan_to_num(trend_prediction, nan=1.0, posinf=1.0, neginf=1.0)
            trend_prediction[np.abs(trend_prediction) < 1e-6] = 1e-6 # Evita divisão por zero
            detrended_sequence = numeric_sequence / trend_prediction
            # Adicione esta linha para limpar quaisquer valores inválidos restantes
            detrended_sequence = np.nan_to_num(detrended_sequence, nan=0.0, posinf=0.0, neginf=0.0)
            
            print("\n--- [Fase 2.3] 🎶 RhythmFinder está a analisar a oscilação...")
            rhythm_brain = ('add', 'sub', 'mul', 'sin', 'cos')
            oscillation_model = find_formula_specialist(n, detrended_sequence, rhythm_brain)
            results['oscillation_formula'] = str(oscillation_model._program)
            
            print("\n" + "-"*40); print("🏆 SÍNTESE FINAL DA ANÁLISE PROFUNDA 🏆"); print("-"*40)
            print(f"📈 Hipótese de Tendência (T): {results['trend_formula']}")
            print(f"🌊 Hipótese de Oscilação (O): {results['oscillation_formula']}")
            
            final_prediction = trend_model.predict(n) * oscillation_model.predict(n)
            final_error = np.mean(np.abs(numeric_sequence - final_prediction))
            results['final_error'] = f"{final_error:.6f}"
            print(f"\n📊 Erro (Fitness) da Fórmula Final: {results['final_error']}")
    else:
        print("\nSinal classificado como NATURAL. Análise profunda não necessária.")
    
    # --- FASE 3: REGISTRO NA BIBLIOTECA ---
    log_results_to_library(results)
    
    print("\n" + "="*60); print("🏛️  ANÁLISE DO ARQUITETO CONCLUÍDA 🏛️"); print("="*60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Análise em duas fases: Classificação e Regressão Simbólica Colaborativa.")
    parser.add_argument("signal_file", type=str, help="Caminho para o arquivo de sinal (.txt).")
    args = parser.parse_args()
    
    run_architect_analysis(args.signal_file)