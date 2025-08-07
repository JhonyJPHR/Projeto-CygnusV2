# Arquivo: src/generators/generate_crab_pulsar_signal.py
import numpy as np
import argparse
import os

def generate_crab_pulsar_spectrogram(time_steps=4096, freq_channels=64):
    """
    Gera um espectrograma 2D que simula as características do Pulsar do Caranguejo.
    """
    # 1. Cria um fundo de ruído
    spectrogram = np.random.normal(loc=0.0, scale=0.5, size=(time_steps, freq_channels))

    # 2. Define as características do pulsar
    # O pulsar real pisca 30x por segundo. Vamos simular essa alta frequência de pulsos.
    # Em 4096 passos de tempo, teremos dezenas de pulsos.
    pulse_period_steps = 30  # Um pulso a cada 30 passos de tempo para simular alta frequência
    pulse_width_steps = 3    # Pulsos muito curtos e nítidos
    pulse_intensity = 10.0   # Pulsos fortes acima do ruído

    # 3. Insere os pulsos no espectrograma
    for t in range(0, time_steps, pulse_period_steps):
        # O pulso é uma linha vertical no espectrograma (acontece em todas as frequências)
        start_time = t
        end_time = t + pulse_width_steps
        if end_time < time_steps:
            spectrogram[start_time:end_time, :] += pulse_intensity

    return spectrogram

def convert_spectrogram_to_binary(spectrogram, n_bits):
    """
    Converte um espectrograma 2D para um sinal binário 1D.
    """
    time_series = spectrogram.mean(axis=1)
    threshold = np.mean(time_series) + 2 * np.std(time_series)
    binary_signal = (time_series > threshold).astype(int)
    
    # Padroniza o comprimento
    if len(binary_signal) > n_bits:
        binary_signal = binary_signal[:n_bits]
    else:
        padding = np.zeros(n_bits - len(binary_signal), dtype=int)
        binary_signal = np.concatenate([binary_signal, padding])
        
    return "".join(map(str, binary_signal))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gera um sinal binário simulado do Pulsar do Caranguejo.')
    parser.add_argument("--length", type=int, default=4096, help="Comprimento final do sinal binário.")
    parser.add_argument("--output_file", type=str, default="data/sinal_crab_pulsar_4096.txt", help="Arquivo de saída para o sinal.")
    args = parser.parse_args()

    print("Gerando espectrograma simulado do 'Pulsar do Caranguejo'...")
    pulsar_spectrogram = generate_crab_pulsar_spectrogram(time_steps=args.length)
    
    print("Convertendo espectrograma para sinal binário...")
    pulsar_binary_signal = convert_spectrogram_to_binary(pulsar_spectrogram, args.length)
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        f.write(pulsar_binary_signal)
        
    print(f"✅ Sinal do Pulsar do Caranguejo simulado e salvo com sucesso em: {args.output_file}")