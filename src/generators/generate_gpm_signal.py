# Arquivo: src/generators/generate_gpm_signal.py
import numpy as np
import argparse
import os

# --- Parâmetros Baseados em GPM J1839–10 ---
# Período real: ~1318 segundos. Para nosso sinal de 4096 amostras,
# vamos simular alguns pulsos dentro desse comprimento.
# Vamos definir um período de ~1024 amostras para garantir ~4 pulsos.
PULSE_PERIOD = 1024
PULSE_WIDTH = 20.0  # Largura do pulso
PULSE_AMPLITUDE = 3.0 # Amplitude do pulso acima do ruído
NOISE_LEVEL = 0.5   # Nível do ruído de fundo

def generate_gpm_signal(length):
    """
    Gera um sinal simulado com as características de um repetidor de longo
    período como o GPM J1839–10.
    """
    # Cria o ruído de fundo
    signal = np.random.randn(length) * NOISE_LEVEL
    
    # Calcula o número de pulsos que cabem no sinal
    num_pulses = length // PULSE_PERIOD
    
    # Adiciona os pulsos em intervalos regulares
    for i in range(1, num_pulses + 1):
        pulse_position = i * PULSE_PERIOD
        # Cria um pulso Gaussiano
        pulse = PULSE_AMPLITUDE * np.exp(-((np.arange(length) - pulse_position)**2) / (2 * PULSE_WIDTH**2))
        signal += pulse
        
    # Converte o sinal analógico para um fluxo binário
    # O limiar (threshold) é um pouco acima do nível de ruído
    threshold = NOISE_LEVEL * 2.0 
    binary_signal = (signal > threshold).astype(int)
    
    return "".join(map(str, binary_signal))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera um sinal simulado de GPM J1839–10.")
    parser.add_argument("--length", type=int, default=4096, help="Comprimento do sinal a ser gerado.")
    parser.add_argument("--output_file", type=str, default="data/sinal_gpm_j1839-10.txt", help="Arquivo de saída para o sinal.")
    args = parser.parse_args()

    print(f"Gerando sinal simulado de GPM J1839–10 com {args.length} amostras...")
    
    # Cria a pasta 'data' se ela não existir
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    gpm_signal_str = generate_gpm_signal(args.length)
    
    with open(args.output_file, "w") as f:
        f.write(gpm_signal_str)
        
    print(f"✅ Sinal salvo com sucesso em: {args.output_file}")