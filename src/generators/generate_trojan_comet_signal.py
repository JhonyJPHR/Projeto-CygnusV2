# Arquivo: src/generators/generate_trojan_comet_signal.py
import numpy as np
import argparse
import os

def generate_trojan_comet_spectrogram(time_steps=4096, freq_channels=64):
    """
    Gera um espectrograma 2D que simula um cometa com um sinal artificial oculto.
    """
    # 1. Cria o ruído de fundo do cometa (ruído rosa/1/f é mais realista para cometas)
    # Para simplificar, usaremos ruído Gaussiano com alguma estrutura
    background_noise = np.random.normal(0, 1.0, (time_steps, freq_channels))
    for _ in range(5): # Adiciona algumas variações de baixa frequência
        idx = np.random.randint(0, time_steps)
        idy = np.random.randint(0, freq_channels)
        background_noise[idx:idx+100, idy:idy+5] += np.random.uniform(-3, 3)

    # 2. Cria o sinal artificial oculto (uma onda senoidal fraca em uma frequência)
    hidden_signal_intensity = 2.5 # Apenas um pouco mais forte que o ruído
    hidden_signal_freq_channel = freq_channels // 3
    
    # Gera a onda senoidal
    time_points = np.arange(time_steps)
    sine_wave = hidden_signal_intensity * np.sin(2 * np.pi * 0.1 * time_points)
    
    # Adiciona a onda senoidal ao espectrograma em um canal de frequência específico
    final_spectrogram = background_noise
    final_spectrogram[:, hidden_signal_freq_channel] += sine_wave

    return final_spectrogram

def convert_spectrogram_to_binary(spectrogram, n_bits):
    """
    Converte o espectrograma final para um sinal binário 1D.
    """
    time_series = spectrogram.mean(axis=1)
    threshold = np.mean(time_series) + 1.5 * np.std(time_series) # Limiar mais sensível
    binary_signal = (time_series > threshold).astype(int)
    
    # Padroniza o comprimento
    if len(binary_signal) > n_bits:
        binary_signal = binary_signal[:n_bits]
    else:
        padding = np.zeros(n_bits - len(binary_signal), dtype=int)
        binary_signal = np.concatenate([binary_signal, padding])
        
    return "".join(map(str, binary_signal))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gera um sinal simulado de um cometa com um sinal artificial oculto.')
    parser.add_argument("--length", type=int, default=4096, help="Comprimento final do sinal binário.")
    parser.add_argument("--output_file", type=str, default="data/sinal_trojan_comet_4096.txt", help="Arquivo de saída para o sinal.")
    args = parser.parse_args()

    print("Gerando espectrograma simulado do cometa 'Cavalo de Troia'...")
    comet_spectrogram = generate_trojan_comet_spectrogram(time_steps=args.length)
    
    print("Convertendo espectrograma para sinal binário...")
    comet_binary_signal = convert_spectrogram_to_binary(comet_spectrogram, args.length)
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        f.write(comet_binary_signal)
        
    print(f"✅ Sinal do cometa 'Cavalo de Troia' salvo com sucesso em: {args.output_file}")