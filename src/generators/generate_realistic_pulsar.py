import numpy as np
import argparse
import os

def generate_dispersed_pulse(time_steps, freq_channels, pulse_time, pulse_width, dm=50):
    """Generates a single, realistic dispersed pulse on a spectrogram."""
    spectrogram = np.zeros((time_steps, freq_channels))
    freqs = np.linspace(1, 0.5, freq_channels)  # Frequências de alta para baixa
    
    # Fórmula simplificada do atraso por dispersão
    time_delay = dm * (freqs**-2 - 1) * (time_steps / 20)

    for i, f in enumerate(freqs):
        # Calcula a posição do pulso no tempo para esta frequência
        dispersed_time = pulse_time + time_delay[i]
        
        # Cria o pulso Gaussiano no tempo
        time_profile = np.exp(-0.5 * ((np.arange(time_steps) - dispersed_time) / pulse_width)**2)
        spectrogram[:, i] += time_profile
        
    return spectrogram

def generate_realistic_pulsar_spectrogram(time_steps=4096, freq_channels=64):
    """Generates a spectrogram with multiple, repeating dispersed pulses."""
    background_noise = np.random.normal(0, 0.5, (time_steps, freq_channels))
    final_spectrogram = background_noise
    
    pulse_period = 150
    pulse_width = 3.0
    
    for t_start in range(pulse_period, time_steps, pulse_period):
        pulse = generate_dispersed_pulse(time_steps, freq_channels, t_start, pulse_width)
        final_spectrogram += pulse * 15 # Aumenta a intensidade do pulso
        
    return final_spectrogram

def convert_spectrogram_to_binary(spectrogram, n_bits):
    """Converts the final spectrogram to a 1D binary signal."""
    time_series = spectrogram.mean(axis=1)
    threshold = np.mean(time_series) + 2.5 * np.std(time_series)
    binary_signal = (time_series > threshold).astype(int)
    
    # Padroniza o comprimento
    binary_signal = binary_signal[:n_bits] if len(binary_signal) > n_bits else np.pad(binary_signal, (0, n_bits - len(binary_signal)))
    return "".join(map(str, binary_signal))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gera um sinal binário simulado de um pulsar realista com dispersão.')
    parser.add_argument("--length", type=int, default=4096, help="Comprimento final do sinal binário.")
    parser.add_argument("--output_file", type=str, default="data/sinal_realistic_pulsar_4096.txt", help="Arquivo de saída.")
    args = parser.parse_args()

    print("Gerando espectrograma de pulsar realista com dispersão...")
    pulsar_spectrogram = generate_realistic_pulsar_spectrogram(time_steps=args.length)
    
    print("Convertendo para sinal binário...")
    pulsar_binary_signal = convert_spectrogram_to_binary(pulsar_spectrogram, args.length)
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        f.write(pulsar_binary_signal)
        
    print(f"✅ Sinal salvo com sucesso em: {args.output_file}")