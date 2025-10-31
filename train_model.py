import os
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import numpy as np
from tqdm import tqdm
import soundfile as sf

# ---------------------
# Model: Denoising Autoencoder
# ---------------------
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(513, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 513),
            nn.ReLU()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ---------------------
# Feature Extraction
# ---------------------
def audio_to_magnitude(filepath):
    """
    Loads an audio file and converts it to a magnitude spectrogram.
    """
    y, sr = librosa.load(filepath, sr=None)
    stft = librosa.stft(y, n_fft=1024)
    mag = np.abs(stft).T  # Shape: (frames, 513)
    return mag, sr, y

# ---------------------
# Dataset Loader
# ---------------------
class SpeechDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for loading paired noisy and clean speech audio files.
    """
    def __init__(self, noisy_dir, clean_dir):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir

        # Get only .wav files and sort them
        self.noisy_files = sorted([
            f for f in os.listdir(noisy_dir) if f.endswith('.wav')
        ])
        self.clean_files = sorted([
            f for f in os.listdir(clean_dir) if f.endswith('.wav')
        ])

        # Ensure same filenames exist in both
        self.filenames = list(set(self.noisy_files).intersection(set(self.clean_files)))
        self.filenames.sort()

        if len(self.filenames) == 0:
            raise ValueError("No matching .wav files found in both directories.")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        noisy_path = os.path.join(self.noisy_dir, filename)
        clean_path = os.path.join(self.clean_dir, filename)

        noisy_mag, _, _ = audio_to_magnitude(noisy_path)
        clean_mag, _, _ = audio_to_magnitude(clean_path)

        min_len = min(len(noisy_mag), len(clean_mag))
        noisy_mag = noisy_mag[:min_len]
        clean_mag = clean_mag[:min_len]

        return torch.tensor(noisy_mag, dtype=torch.float32), torch.tensor(clean_mag, dtype=torch.float32)

# ---------------------
# Training
# ---------------------
def train_model():
    """
    Main function to train the Denoising Autoencoder model.
    """
    # Create dataset directories if they don't exist
    for d in ["dataset/noisy_trainset_wav", "dataset/clean_trainset_wav"]:
        os.makedirs(d, exist_ok=True)
    
    try:
        dataset = SpeechDataset("dataset/noisy_trainset_wav", "dataset/clean_trainset_wav")
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    except ValueError as e:
        print(f"Error: {e}")
        print("Please place paired .wav files in 'dataset/noisy_trainset_wav' and 'dataset/clean_trainset_wav'.")
        return

    model = DenoisingAutoencoder()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    num_epochs = 10
    model.train()
    print("🚀 Starting model training...")
    for epoch in range(num_epochs):
        total_loss = 0.0
        for noisy_batch, clean_batch in tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # noisy_batch/clean_batch: (1, frames, 513)
            noisy_batch = noisy_batch.squeeze(0)
            clean_batch = clean_batch.squeeze(0)

            optimizer.zero_grad()
            output = model(noisy_batch)
            loss = criterion(output, clean_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # Save model
    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), "model/denoising_autoencoder.pth")
    print("✅ Model saved to model/denoising_autoencoder.pth")


if __name__ == "__main__":
    train_model()