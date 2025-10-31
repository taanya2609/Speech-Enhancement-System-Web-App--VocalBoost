import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import io
import base64

def plot_spectrogram(audio_path, output_path):
    """
    Generates and saves a spectrogram plot of the audio file.
    """
    y, sr = librosa.load(audio_path, sr=None)
    
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def plot_and_get_base64(audio_path):
    """
    Generates a spectrogram plot and returns it as a base64 encoded string.
    """
    y, sr = librosa.load(audio_path, sr=None)
    
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_base64