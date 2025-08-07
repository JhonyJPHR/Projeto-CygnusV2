# Arquivo: src/generators/generate_wow_signal.py
import numpy as np
import argparse
import os

def generate_wow_signal_spectrogram(time_steps=256, freq_channels=64):
    """
    Gera um espectrograma 2D que simula as características do Sinal Wow!
    """
    # 1. Cria um fundo de ruído
    spectrogram = np.random.normal(loc=0.0, scale=0.5, size=(time_steps, freq_channels))

    # 2. Define as características do sinal
    signal_duration = 72  # O sinal real durou 72s. Vamos mapear isso para nossos time_steps.
    center_time = time_steps // 2
    time_std_dev = signal_duration / 6.0 # Desvio padrão para a curva Gaussiana

    # Perfil de intensidade do sinal (curva de sino/Gaussiana)
    time_profile = np.exp(-0.5 * ((np.arange(time_steps) - center_time) / time_std_dev)**2)
    
    # 3. Insere o sinal no espectrograma
    # O sinal era de banda estreita, então o inserimos em poucos canais de frequência
    start_freq_channel = freq_channels // 2
    end_freq_channel = start_freq_channel + 3 # Apenas 3 canais de largura
    signal_intensity = 30.0 # O sinal real era ~30x mais forte que o ruído de fundo

    for t in range(time_steps):
        spectrogram[t, start_freq_channel:end_freq_channel] += time_profile[t] * signal_intensity

    return spectrogram

def convert_spectrogram_to_binary(spectrogram, n_bits):
    """
    Converte um espectrograma 2D para um sinal binário 1D.
    """
    # Soma a intensidade de todas as frequências para cada instante de tempo
    time_series = spectrogram.mean(axis=1)
    
    # Binariza usando um limiar estatístico
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
    parser = argparse.ArgumentParser(description='Gera um sinal binário simulado do "Sinal Wow!".')
    parser.add_argument("--length", type=int, default=4096, help="Comprimento final do sinal binário.")
    parser.add_argument("--output_file", type=str, default="data/sinal_wow_4096.txt", help="Arquivo de saída para o sinal.")
    args = parser.parse_args()

    print("Gerando espectrograma simulado do 'Sinal Wow!'...")
    # Usamos mais time_steps para gerar o espectrograma e depois o padronizamos para 4096 bits
    wow_spectrogram = generate_wow_signal_spectrogram(time_steps=args.length)
    
    print("Convertendo espectrograma para sinal binário...")
    wow_binary_signal = convert_spectrogram_to_binary(wow_spectrogram, args.length)
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        f.write(wow_binary_signal)
        
    print(f"✅ Sinal 'Wow!' simulado e salvo com sucesso em: {args.output_file}")