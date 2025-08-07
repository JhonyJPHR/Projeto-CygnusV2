# Arquivo: utils_spectrogram.py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# --- Parâmetros de Configuração para os Espectrogramas ---
# Você pode ajustar estes valores para otimizar a resolução em tempo vs. frequência
NPERSEG = 128          # Comprimento de cada segmento da STFT
NOVERLAP = NPERSEG // 2  # Sobreposição entre segmentos (geralmente 50%)
IMG_SIZE = (64, 64)   # Tamanho final da imagem para a IA

def generate_spectrogram(raw_signal, fs=1000):
    """
    Converte um sinal 1D bruto em uma imagem de espectrograma 2D.

    Args:
        raw_signal (np.array): O sinal 1D.
        fs (int): A frequência de amostragem do sinal.

    Returns:
        np.array: Uma imagem de espectrograma normalizada (matriz 2D).
    """
    # Garante que o sinal de entrada seja um array numpy
    raw_signal = np.asarray(raw_signal, dtype=float)

    # Calcula a STFT
    # Zxx é uma matriz complexa onde as linhas são frequências e as colunas são tempo
    f, t, Zxx = signal.stft(raw_signal, fs=fs, nperseg=NPERSEG, noverlap=NOVERLAP)

    # Converte os valores complexos para uma escala de potência (Decibéis)
    # Usamos np.abs para obter a magnitude e 20*log10 para converter para dB
    # Adicionamos um valor pequeno (1e-9) para evitar log(0)
    spectrogram = 20 * np.log10(np.abs(Zxx) + 1e-9)

    # Normaliza a imagem para o intervalo [0, 1] para a rede neural
    min_val = np.min(spectrogram)
    max_val = np.max(spectrogram)
    if max_val > min_val:
        spectrogram = (spectrogram - min_val) / (max_val - min_val)
    else:
        # Caso de sinal constante, o espectrograma será todo preto
        spectrogram = np.zeros_like(spectrogram)

    # Redimensiona a imagem para o tamanho padrão que a IA espera
    # Usaremos uma biblioteca de imagem para um redimensionamento de alta qualidade
    # Por simplicidade aqui, vamos usar a função de redimensionamento do numpy/scipy,
    # mas para produção, `cv2.resize` ou `PIL.Image.resize` são melhores.
    # Nota: para este exemplo, vamos pular o redimensionamento complexo e
    # assumir que os parâmetros da STFT já geram um tamanho aproximado.
    # O ideal é adicionar `skimage.transform.resize` ou `cv2.resize`.
    
    # Simplesmente garantimos que o output tenha a forma esperada (ex: cortando/preenchendo)
    # Esta é uma simplificação. Uma abordagem melhor usaria interpolação.
    h, w = spectrogram.shape
    final_img = np.zeros(IMG_SIZE)
    h_min, w_min = min(h, IMG_SIZE[0]), min(w, IMG_SIZE[1])
    final_img[:h_min, :w_min] = spectrogram[:h_min, :w_min]
    
    return final_img

if __name__ == '__main__':
    print("🚀 Testando o Gerador de Espectrogramas 🚀")
    
    # Cria um sinal de teste: duas ondas senoidais com ruído
    fs = 10000
    t = np.linspace(0, 1, fs, endpoint=False)
    test_signal = np.sin(2 * np.pi * 500 * t)  # Senoide de 500 Hz
    test_signal[fs//2:] += np.sin(2 * np.pi * 1500 * t[fs//2:]) # Adiciona 1500 Hz na metade
    test_signal += np.random.randn(len(t)) * 0.1 # Adiciona ruído

    print("Sinal de teste 1D gerado. Convertendo para Espectrograma 2D...")
    spectrogram_image = generate_spectrogram(test_signal, fs=fs)
    print(f"Espectrograma gerado com sucesso. Dimensões: {spectrogram_image.shape}")

    # Visualiza e salva o espectrograma
    plt.figure(figsize=(8, 6))
    plt.imshow(spectrogram_image, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Intensidade Normalizada')
    plt.title('Espectrograma do Sinal de Teste')
    plt.xlabel('Tempo')
    plt.ylabel('Frequência')
    plt.savefig('espectrograma_exemplo.png')
    plt.show()

    print("✅ Imagem 'espectrograma_exemplo.png' salva.")