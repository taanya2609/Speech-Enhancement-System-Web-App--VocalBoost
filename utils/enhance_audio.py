import torch
import librosa
import numpy as np
from scipy.signal import find_peaks
import os
import soundfile as sf
from train_model import DenoisingAutoencoder

try:
    model = DenoisingAutoencoder()
    model.load_state_dict(torch.load('model/denoising_autoencoder.pth', map_location=torch.device('cpu')))
    model.eval()
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print("❌ Model file 'model/denoising_autoencoder.pth' not found. Please run train_model.py first.")
    model = None

def magnitude_to_audio(mag_spec, sr):
    """
    Converts a magnitude spectrogram back to an audio signal using Griffin-Lim.
    """
    stft_matrix = librosa.db_to_amplitude(librosa.amplitude_to_db(mag_spec.T))
    enhanced_audio = librosa.griffinlim(stft_matrix, n_iter=50, hop_length=256)
    return enhanced_audio

def enhance_audio(input_path, output_path, level="normal"):
    """
    Processes an input audio file to enhance it using the loaded model.
    """
    if model is None:
        return None, None, None

    y, sr = librosa.load(input_path, sr=None)
    stft = librosa.stft(y, n_fft=1024)
    mag_spec = np.abs(stft)
    phase_spec = np.angle(stft)

    input_tensor = torch.tensor(mag_spec.T, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        enhanced_mag = model(input_tensor).squeeze(0)

    # Apply enhancement level based on user selection
    enhancement_factor = 1.0
    if level == "high":
        enhancement_factor = 1.3
    
    enhanced_mag *= enhancement_factor
    
    enhanced_stft = enhanced_mag.T.numpy() * np.exp(1j * phase_spec)
    
    enhanced_audio = librosa.istft(enhanced_stft, hop_length=256)
    
    confidence = calculate_confidence(y, enhanced_audio, sr)
    
    sf.write(output_path, enhanced_audio, sr)

    return output_path, sr, confidence

def calculate_confidence(original_audio, enhanced_audio, sr):
    """
    Estimates a confidence score based on signal quality metrics.
    A higher score indicates better enhancement.
    """
    original_spectrogram = librosa.feature.melspectrogram(y=original_audio, sr=sr)
    enhanced_spectrogram = librosa.feature.melspectrogram(y=enhanced_audio, sr=sr)
    
    original_peaks, _ = find_peaks(original_spectrogram.flatten(), height=np.mean(original_spectrogram)*1.5)
    enhanced_peaks, _ = find_peaks(enhanced_spectrogram.flatten(), height=np.mean(enhanced_spectrogram)*1.5)
    
    if len(original_peaks) > 0:
        score = min(len(enhanced_peaks) / len(original_peaks), 1.0)
    else:
        score = 0.5
    
    confidence_score = int(score * 100)
    return confidence_score