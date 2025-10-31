### **VocalBoost: Speech Enhancement System**

This project enhances speech clarity by removing background noise using a Denoising Autoencoder (DAE) model. It provides a real-time, web-based platform where users can upload or record audio, process it through a deep learning model, and receive improved, noise-free output with spectrogram comparison.

---

### **Project Overview**

**Title:**
VocalBoost: AI-Based Speech Enhancement Using Deep Learning

**Objective:**
To design a deep learning–driven speech enhancement system that removes background noise from audio using a Denoising Autoencoder, integrated with a web interface for real-time enhancement, playback, and visualization.

---

### **Dataset Details:**

* Source: Datashare Edinburgh (Valentini-Botinhao, 2017)
* Trainset: 11,572 noisy-clean pairs
* Testset: 824 noisy-clean pairs
* Format: `.wav` files organized into noisy and clean directories

---
### **Algorithm And Process Design:**

* Model Type: Denoising Autoencoder (Deep Generative Model)
* Approach: Speech Enhancement using Time-Frequency Domain Features (STFT)
* Algorithm Used: Neural Network with Encoder–Decoder Architecture
* Regression Type: Non-linear mapping (signal-to-signal regression)
* Libraries Used: PyTorch, Librosa, NumPy, Matplotlib, Flask
  
---
### **Details Of Hardware And Software:**

* Hardware: Computer/Server with multi-core CPU, GPU (NVIDIA CUDA supported) for deep learning model training, minimum 8GB RAM (16GB+ recommended), SSD for dataset and model storage, and microphone & speakers for recording and playback.
* Software Requirements
Programming Language: Python
Frameworks & Libraries:
PyTorch – for training the Denoising Autoencoder model
Librosa – for audio feature extraction (spectrograms)
FFmpeg – for audio conversion
Flask – for backend web framework
JavaScript (with Web Audio API) – for client-side recording & visualization
HTML, CSS – for frontend design
Matplotlib – for spectrogram visualization

---
### **Key Components:**

**1. Dataset Preparation**
Paired noisy and clean audio files enable supervised learning, helping the model learn mappings between distorted and clean speech.

**2. Feature Extraction**
Raw audio converted into magnitude spectrograms using Librosa, providing time–frequency representation for model input.

**3. Model Architecture**
A **Denoising Autoencoder (DAE)** removes noise by minimizing Mean Squared Error (MSE) between predicted and clean spectrograms.

* Trained using `train_model.py`
* Model saved as `model/denoising_autoencoder.pth`

**4. Web Application Framework**

* **Frontend:** HTML, CSS, JavaScript (`recorder.js` for mic input)
* **Backend:** Flask (Python)
* **Audio Conversion:** FFmpeg standardizes all files to `.wav` format

**5. Enhancement & Reconstruction**
`enhance_audio.py` loads the trained model, performs denoising, and reconstructs clean audio using the **Griffin-Lim algorithm**.
Includes confidence scoring and spectrogram visualization using `plot_spectrogram.py`.

---

### **Results Summary:**

* Significant reduction in background noise and improved speech clarity
* Accurate reconstruction of clean signals
* Real-time enhancement with visual waveform and spectrogram comparison
* Clear audio files cann be downloaded by user

---

### **Applications:**

* Voice communication systems
* Call centers and conferencing tools
* Hearing assistive devices
* Podcast and studio audio cleaning

---

### **Future Enhancements:**

* Real-time Denoising: Enable live noise suppression during calls and meetings.
* Mobile App Integration: Extend the service to Android/iOS platforms.
* Multi-Language Support: Optimize for speech recognition and transcription systems.
* Adaptive Learning: Improve accuracy by retraining on user feedback.
* Cloud Deployment: Scale the system for large user bases with faster processing.

---
