import cv2
import numpy as np
import tensorflow as tf
import os
import time
import matplotlib.pyplot as plt
from pytube import YouTube

analysis_data = {}
degradation_records = []
engagement_data = {}  # Variável global para armazenar dados de engajamento

def download_youtube_video(url, path):
    yt = YouTube(url)
    stream = yt.streams.filter(file_extension='mp4').first()
    stream.download(filename=path)

def check_video_integrity(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    ret, _ = cap.read()
    cap.release()
    return ret

# Função para calcular o SSIM usando TensorFlow para imagens em cores
def calculate_ssim(img1, img2):
    img1 = tf.convert_to_tensor(img1, dtype=tf.float32)
    img2 = tf.convert_to_tensor(img2, dtype=tf.float32)
    ssim = tf.image.ssim(img1, img2, max_val=255)
    return tf.reduce_mean(ssim).numpy()

# Função para calcular o PSNR usando TensorFlow para imagens em cores
def calculate_psnr(img1, img2):
    img1 = tf.convert_to_tensor(img1, dtype=tf.float32)
    img2 = tf.convert_to_tensor(img2, dtype=tf.float32)
    mse = tf.reduce_mean(tf.square(img1 - img2))
    psnr = 20 * tf.math.log(255.0 / tf.sqrt(mse)) / tf.math.log(10.0)
    return psnr.numpy()

# Função para calcular o MSE
def calculate_mse(img1, img2):
    img1 = np.array(img1, dtype=np.float32)
    img2 = np.array(img2, dtype=np.float32)
    mse = np.mean((img1 - img2) ** 2)
    return mse

# Função para calcular o VQM (aproximado)
def calculate_vqm(frames):
    ssim_values = [calculate_ssim(frames[i-1], frames[i]) for i in range(1, len(frames))]
    psnr_values = [calculate_psnr(frames[i-1], frames[i]) for i in range(1, len(frames))]
    vqm_value = (1 - np.mean(ssim_values)) * 0.5 + (30 - np.mean(psnr_values)) * 0.5
    return vqm_value

# Função para calcular o Dropout Rate
def calculate_dropout_rate(total_frames, dropped_frames):
    if total_frames > 0:
        return dropped_frames / total_frames
    else:
        return 0

# Função para calcular a taxa de buffering
def calculate_buffering_rate(frame_times):
    if len(frame_times) < 2:
        return 0
    else:
        buffer_times = [frame_times[i] - frame_times[i-1] for i in range(1, len(frame_times))]
        avg_buffer_time = np.mean(buffer_times)
        if avg_buffer_time > 0:
            return 1 / avg_buffer_time
        else:
            return 0

# Função para calcular a "acurácia" das métricas em comparação com os limiares
def calculate_accuracy(metric_value, threshold):
    if metric_value >= threshold:
        return 1.0  # Considerar como "acertado" se o valor estiver acima do limiar
    else:
        return metric_value / threshold  # Proporção do valor em relação ao limiar

def convert_to_serializable(data):
    if isinstance(data, dict):
        return {k: convert_to_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(i) for i in data]
    elif isinstance(data, np.float32):
        return float(data)
    return data

def plot_metrics(time_stamps, ssim_values, psnr_values, mse_values, vqm_values, ssim_accuracy, psnr_accuracy, mse_accuracy, vqm_accuracy):
    plt.figure(figsize=(24, 12))

    plt.subplot(2, 4, 1)
    plt.plot(time_stamps, ssim_values, label='SSIM', color='b')
    plt.axhline(y=0.95, color='r', linestyle='--', label='Limiar SSIM')
    plt.xlabel('Tempo (segundos)')
    plt.ylabel('SSIM')
    plt.title('Variação de SSIM ao longo do tempo')
    plt.legend()

    plt.subplot(2, 4, 2)
    plt.plot(time_stamps, psnr_values, label='PSNR', color='g')
    plt.axhline(y=30.0, color='r', linestyle='--', label='Limiar PSNR')
    plt.xlabel('Tempo (segundos)')
    plt.ylabel('PSNR (dB)')
    plt.title('Variação de PSNR ao longo do tempo')
    plt.legend()

    plt.subplot(2, 4, 3)
    plt.plot(time_stamps, mse_values, label='MSE', color='m')
    plt.axhline(y=30.0, color='r', linestyle='--', label='Limiar MSE')
    plt.xlabel('Tempo (segundos)')
    plt.ylabel('MSE')
    plt.title('Variação de MSE ao longo do tempo')
    plt.legend()

    plt.subplot(2, 4, 4)
    plt.plot(time_stamps, vqm_values, label='VQM', color='c')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Limiar VQM')
    plt.xlabel('Tempo (segundos)')
    plt.ylabel('VQM')
    plt.title('Variação de VQM ao longo do tempo')
    plt.legend()

    plt.subplot(2, 4, 5)
    plt.plot(time_stamps, ssim_accuracy, label='Acurácia SSIM', color='b')
    plt.xlabel('Tempo (segundos)')
    plt.ylabel('Acurácia')
    plt.title('Acurácia de SSIM ao longo do tempo')
    plt.legend()

    plt.subplot(2, 4, 6)
    plt.plot(time_stamps, psnr_accuracy, label='Acurácia PSNR', color='g')
    plt.xlabel('Tempo (segundos)')
    plt.ylabel('Acurácia')
    plt.title('Acurácia de PSNR ao longo do tempo')
    plt.legend()

    plt.subplot(2, 4, 7)
    plt.plot(time_stamps, mse_accuracy, label='Acurácia MSE', color='m')
    plt.xlabel('Tempo (segundos)')
    plt.ylabel('Acurácia')
    plt.title('Acurácia de MSE ao longo do tempo')
    plt.legend()

    plt.subplot(2, 4, 8)
    plt.plot(time_stamps, vqm_accuracy, label='Acurácia VQM', color='c')
    plt.xlabel('Tempo (segundos)')
    plt.ylabel('Acurácia')
    plt.title('Acurácia de VQM ao longo do tempo')
    plt.legend()

    plt.tight_layout()
    plt.savefig('static/metrics_plot.png')
    plt.close()

def save_frame(frame, timestamp, metric_values):
    frame_filename = f"static/degradation_frame_{timestamp:.2f}.png"
    cv2.imwrite(frame_filename, frame)
    degradation_records.append({
        'timestamp': timestamp,
        'ssim': metric_values['ssim'],
        'psnr': metric_values['psnr'],
        'mse': metric_values['mse'],
        'vqm': metric_values['vqm'],
        'frame': frame_filename
    })

def analyze_video(video_path):
    global analysis_data, degradation_records, engagement_data
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return {}

    resolution = f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"

    frame_count = 0
    dropped_frames = 0
    frames_buffer = []
    frame_times = []
    ssim_values_over_time = []
    psnr_values_over_time = []
    mse_values_over_time = []
    vqm_values_over_time = []
    ssim_accuracy_over_time = []
    psnr_accuracy_over_time = []
    mse_accuracy_over_time = []
    vqm_accuracy_over_time = []
    time_stamps = []

    window_size = 5
    degradation_thresholds = {
        'ssim': 0.95,
        'psnr': 30.0,
        'mse': 30.0,
        'vqm': 0.5
    }

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total de frames no vídeo
    fps = cap.get(cv2.CAP_PROP_FPS) or 1  # FPS do vídeo
    video_duration_seconds = total_frames / fps  # Duração total do vídeo em segundos

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frames_buffer.append(frame)
        if len(frames_buffer) > window_size:
            frames_buffer.pop(0)

        if len(frames_buffer) == window_size:
            ssim_values = []
            psnr_values = []
            mse_values = []
            for i in range(1, window_size):
                ssim_values.append(calculate_ssim(frames_buffer[i-1], frames_buffer[i]))
                psnr_values.append(calculate_psnr(frames_buffer[i-1], frames_buffer[i]))
                mse_values.append(calculate_mse(frames_buffer[i-1], frames_buffer[i]))

            avg_ssim = np.mean(ssim_values)
            avg_psnr = np.mean(psnr_values)
            avg_mse = np.mean(mse_values)
            avg_vqm = calculate_vqm(frames_buffer)

            ssim_values_over_time.append(avg_ssim)
            psnr_values_over_time.append(avg_psnr)
            mse_values_over_time.append(avg_mse)
            vqm_values_over_time.append(avg_vqm)
            timestamp = frame_count / fps
            time_stamps.append(timestamp)

            ssim_accuracy_over_time.append(calculate_accuracy(avg_ssim, degradation_thresholds['ssim']))
            psnr_accuracy_over_time.append(calculate_accuracy(avg_psnr, degradation_thresholds['psnr']))
            mse_accuracy_over_time.append(calculate_accuracy(degradation_thresholds['mse'], avg_mse))  # Note que para MSE, queremos que o valor seja baixo
            vqm_accuracy_over_time.append(calculate_accuracy(degradation_thresholds['vqm'], avg_vqm))  # Note que para VQM, queremos que o valor seja baixo

            if avg_ssim < degradation_thresholds['ssim'] or avg_psnr < degradation_thresholds['psnr'] or avg_mse > degradation_thresholds['mse'] or avg_vqm > degradation_thresholds['vqm']:
                save_frame(frame, timestamp, {'ssim': avg_ssim, 'psnr': avg_psnr, 'mse': avg_mse, 'vqm': avg_vqm})

        frame_count += 1
        frame_times.append(time.time())  # Registro do tempo atual
        analysis_data['progress'] = round((frame_count / total_frames) * 100)  # Atualizar progresso
        time.sleep(0.1)  # Adiciona um pequeno atraso para simular um processamento mais lento e permitir a atualização da barra de progresso

    cap.release()

    file_size_bits = os.path.getsize(video_path) * 8

    if video_duration_seconds > 0:
        avg_bitrate = (file_size_bits / 1000) / video_duration_seconds
    else:
        avg_bitrate = 0

    dropout_rate = calculate_dropout_rate(frame_count, dropped_frames)
    buffering_rate = calculate_buffering_rate(frame_times)

    # Recuperar dados de engajamento
    engagement = engagement_data.get(video_path, {'watched_time': 0, 'total_duration': video_duration_seconds})
    watched_time = engagement['watched_time']
    total_duration = engagement['total_duration']

    if total_duration > 0:
        engagement_ratio = watched_time / total_duration
    else:
        engagement_ratio = 0  # Evitar divisão por zero

    analysis_data = {
        'ssim': float(np.mean(ssim_values_over_time)),
        'psnr': float(np.mean(psnr_values_over_time)),
        'mse': float(np.mean(mse_values_over_time)),
        'vqm': float(np.mean(vqm_values_over_time)),
        'ssim_accuracy': float(np.mean(ssim_accuracy_over_time)),
        'psnr_accuracy': float(np.mean(psnr_accuracy_over_time)),
        'mse_accuracy': float(np.mean(mse_accuracy_over_time)),
        'vqm_accuracy': float(np.mean(vqm_accuracy_over_time)),
        'resolution': resolution,
        'bitrate': f"{avg_bitrate:.2f} kbps",
        'compression': 'Moderado',
        'frames': degradation_records,
        'dropout_rate': dropout_rate,
        'buffering_rate': buffering_rate,
        'engagement': float(np.float32(0.85)),
        'analysis_complete': True,
        'progress': 100  # Análise completa
    }

    plot_metrics(time_stamps, ssim_values_over_time, psnr_values_over_time, mse_values_over_time, vqm_values_over_time, ssim_accuracy_over_time, psnr_accuracy_over_time, mse_accuracy_over_time, vqm_accuracy_over_time)
    analysis_data = convert_to_serializable(analysis_data)
    degradation_records = []
    return analysis_data
